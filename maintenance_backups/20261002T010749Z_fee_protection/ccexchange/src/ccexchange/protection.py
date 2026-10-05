"""Position protection independent of indicator downloads. No entry strategy here."""
import json
import logging
import math
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd

from .execution import OrderIntent, order_id
from .risk import ratchet_stop
from .state import ManagedPosition

LOG = logging.getLogger(__name__)
TERMINAL = {'filled', 'rejected', 'canceled', 'expired'}


def emit(kind, managed, now, *, path=Path('logs/monitor_events.jsonl'), **extra):
    """Optional telemetry is never allowed to interrupt protection/order execution."""
    try:
        event = dict(schema_version=1, event_id=str(uuid.uuid4()), bot_id='ccexchange',
                     strategy='crypto_momentum', trade_id=managed.trade_id,
                     timestamp=now.isoformat(), symbol=managed.symbol, event_type=kind,
                     provenance='RECORDED', source='CCExchange position protection',
                     risk={'stop_id':'atr','label':'ATR trailing stop' if managed.atr>0 else 'Saved protective stop','price_basis':'underlying',
                           'current_stop':managed.stop,'atr':managed.atr,'atr_as_of':managed.atr_as_of},
                     position={'highest_price':managed.high_watermark,'quantity':managed.quantity}, **extra)
        if managed.entry_order_id and 'order_id' not in event:
            event['order_id'] = managed.entry_order_id
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a') as handle:
            handle.write(json.dumps(event, allow_nan=False)+'\n')
        return True
    except Exception:
        LOG.exception('Optional stop telemetry failed')
        return False


def fresh_quote(quote, now):
    price, ask = quote['price'], quote['ask']
    stamp = pd.Timestamp(quote['timestamp'])
    if stamp.tzinfo is None or not all(math.isfinite(v) and v > 0 for v in (price, ask)) or ask < price:
        raise ValueError('Invalid quote')
    age = (now-stamp.to_pydatetime(warn=False)).total_seconds()
    if not -5 <= age <= 60:
        raise ValueError('Stale or future quote')
    return price


def submit_exit(symbol, reason, price, state, store, broker, cfg, audit, now, actual, paper=False):
    """Persist before POST; ambiguous responses never authorize another POST."""
    managed = state.positions[symbol]
    pending = state.exit_orders.get(symbol)
    if pending:
        if pending.get('phase') == 'retry':
            if now < datetime.fromisoformat(pending['retry_after']):
                return
        else:
            audit.write('EXIT_PENDING', symbol=symbol, client_order_id=pending['client_id'])
            return
    elif state.pending_orders.get(symbol) == 'sell':
        # Legacy pending order has no durable client ID. Do not guess its outcome.
        audit.write('EXIT_REQUIRES_RECONCILIATION',symbol=symbol,reason='Legacy pending sell has no durable identity')
        return
    cid = 'ccexchange-'+uuid.uuid4().hex
    state.exit_orders[symbol] = {'client_id':cid,'phase':'submitting','reason':reason,'submitted_at':now.isoformat()}
    state.pending_orders[symbol] = 'sell'
    state.cooldowns[symbol] = (cfg.risk.loss_cooldown_bars if price < managed.entry_price else cfg.risk.symbol_cooldown_bars)+1
    store.save(state)  # If persistence fails, no broker submission occurs.
    emit('EXIT_DECISION',managed,now,decision='SELL',reason=reason,market={'price':price})
    try:
        result = broker.submit(OrderIntent(symbol,'sell',actual.quantity,reason,cid))
        state.exit_orders[symbol].update(phase='submitted',order_id=order_id(result))
        store.save(state)
        audit.write('SELL_SUBMITTED',symbol=symbol,order_id=order_id(result),client_order_id=cid,
                    reason=reason,stop=managed.stop,current=price,entry=managed.entry_price,quantity=actual.quantity)
        emit('EXIT_ORDER_SUBMITTED',managed,now,order_id=order_id(result),reason=reason,market={'price':price})
        try:
            from .paper import PaperRecorder
            if paper: PaperRecorder().record_order({'order_id':order_id(result),'timestamp':now.isoformat(),
                'symbol':symbol,'side':'sell','reason':reason,'client_order_id':cid,
                'quantity':actual.quantity,'signal_price':price,'submitted_at':now.isoformat(),
                'experiment_id':managed.experiment_id,'timeframe':managed.timeframe})
        except Exception:
            LOG.exception('Optional exit analytics failed')
    except Exception:
        # The request may have reached the broker. Recovery only looks up this ID.
        audit.write('EXIT_SUBMISSION_UNCERTAIN',symbol=symbol,client_order_id=cid)
        LOG.exception('Exit submission requires broker reconciliation for %s',symbol)


def reconcile_exits(state, store, broker, audit, now):
    for symbol, pending in list(state.exit_orders.items()):
        if pending.get('phase')=='retry':
            continue
        try:
            order = broker.lookup_order(pending['client_id'])
            if order is None:
                state.protection_issues[symbol]='Exit submission outcome unresolved'
                audit.write('EXIT_UNRESOLVED',symbol=symbol,client_order_id=pending['client_id'],reason='Order lookup returned no record; no duplicate submitted')
                continue
            status=order['status']
            managed=state.positions.get(symbol)
            if status!=pending.get('status'):
                if managed:
                    kind={'filled':'EXIT_FILLED','rejected':'ORDER_REJECTED','canceled':'ORDER_CANCELLED','expired':'ORDER_EXPIRED'}.get(status,'ORDER_ACCEPTED')
                    emit(kind,managed,now,order_id=order['id'],order_status=status,filled_quantity=order['filled_qty'])
                pending['status']=status
                audit.write('EXIT_ORDER_STATUS',symbol=symbol,status=status,filled_quantity=order['filled_qty'])
            if status in TERMINAL:
                pending.update(phase='retry',retry_after=(now+timedelta(seconds=60)).isoformat())
                # Next pass obtains fresh holdings before any remainder is retried.
            elif (now-datetime.fromisoformat(pending['submitted_at'])).total_seconds()>120:
                state.protection_issues[symbol]='Exit order accepted but not completed within 120 seconds'
                audit.write('EXIT_ORDER_STALLED',symbol=symbol,status=status,client_order_id=pending['client_id'])
            store.save(state)
        except Exception:
            state.protection_issues[symbol]='Exit reconciliation unavailable'
            audit.write('EXIT_RECONCILIATION_UNAVAILABLE',symbol=symbol,client_order_id=pending['client_id'])
            LOG.exception('Unable to reconcile exit for %s',symbol)


def manage_positions(state, store, broker, cfg, audit, now, account, rows=None, regime=None, dry_run=False, paper=False):
    rows=rows or {}
    state.protection_issues = {}
    reconcile_exits(state,store,broker,audit,now)
    for symbol, actual in account.positions.items():
        if symbol not in cfg.symbols or actual.quantity <= 0:
            continue
        if symbol not in state.positions:
            # Lost/missing local history: preserve emergency protection, never invent ATR.
            state.positions[symbol]=ManagedPosition(symbol,actual.average_entry,actual.quantity,
                actual.average_entry*(1-cfg.risk.emergency_stop_percent),actual.average_entry,now.isoformat(),
                trade_id='adopted-'+uuid.uuid4().hex)
            audit.write('PROTECTION_ADOPTED',symbol=symbol,reason='Historical trailing stop unavailable; emergency floor only')
        managed=state.positions[symbol]
        if not managed.trade_id:
            managed.trade_id='position-'+uuid.uuid4().hex
        managed.quantity=actual.quantity
        managed.entry_price=actual.average_entry
        try:
            quote=broker.quote(symbol)
            price=fresh_quote(quote,datetime.now(timezone.utc))
        except Exception:
            state.protection_issues[symbol]='Fresh quote unavailable; saved stop retained'
            audit.write('PROTECTION_QUOTE_UNAVAILABLE',symbol=symbol,reason='Fresh timestamped bid required; saved stop retained')
            continue
        managed.high_watermark=max(managed.high_watermark,price)
        row=rows.get(symbol)
        if row is not None and math.isfinite(float(row.atr)) and float(row.atr)>0:
            managed.atr=float(row.atr)
            managed.atr_as_of=now.isoformat()
            managed.high_watermark=max(managed.high_watermark,float(row.high))
        if managed.atr>0:
            managed.stop=ratchet_stop(managed.stop,managed.high_watermark,managed.atr,cfg.risk)
        store.save(state)
        if managed.reported_stop != managed.stop:
            recorded=emit('STOP_UPDATED',managed,now,market={'price':price,'timestamp':quote['timestamp']},
                 reason='Recorded trailing threshold; ATR is last successfully calculated value')
            if recorded:
                managed.reported_stop=managed.stop
            store.save(state)
        reason=None
        if price<=managed.stop:reason='ratcheting ATR stop' if managed.atr>0 else 'saved protective stop'
        elif price<=managed.entry_price*(1-cfg.risk.emergency_stop_percent):reason='emergency loss stop'
        elif regime is not None and getattr(regime,'value',regime)=='HIGH-RISK / CRASH':reason='high-risk market regime'
        elif row is not None and row.macd<row.macd_signal and row.close<row.ema_fast:reason='MACD and EMA momentum deterioration'
        pending=state.exit_orders.get(symbol)
        # Once an exit was triggered, rejected/canceled/partial remainders remain exits even after a rebound.
        if pending and pending.get('phase')=='retry':reason=pending['reason']
        if reason:
            if state.pending_orders.get(symbol)=='sell' and symbol not in state.exit_orders:
                state.protection_issues[symbol]='Legacy pending sell requires reconciliation'
            if dry_run:
                audit.write('DRY_RUN_EXIT',symbol=symbol,reason=reason,stop=managed.stop,price=price)
            else:submit_exit(symbol,reason,price,state,store,broker,cfg,audit,now,actual,paper)
    for symbol in list(state.positions):
        if symbol in account.positions or state.pending_orders.get(symbol)=='buy':
            continue
        pending=state.exit_orders.get(symbol)
        if pending and pending.get('phase')!='retry':
            continue  # Broker response/position consistency must be established first.
        managed=state.positions.pop(symbol)
        emit('POSITION_CLOSED',managed,now,reason='Broker confirms no owned position')
        state.exit_orders.pop(symbol,None);state.pending_orders.pop(symbol,None)
    store.save(state)


def protect_cycle(settings, cfg, audit, broker, store, now=None):
    now=now or datetime.now(timezone.utc)
    with store.locked():
        state=store.load()
        account=broker.account()
        manage_positions(state,store,broker,cfg,audit,now,account,dry_run=settings.dry_run,paper=getattr(settings,'paper_trading',False))
    # Separate bounded heartbeat; stale/failed quote checks are explicitly reported in audit.
    try:
        path=store.path.with_name('protection_health.json')
        temp=path.with_suffix('.tmp')
        temp.write_text(json.dumps({'checked_at':datetime.now(timezone.utc).isoformat(),'positions':len(account.positions),'pending_exits':len(state.exit_orders),'issues':state.protection_issues}))
        temp.replace(path)
    except Exception:
        LOG.exception('Optional protection heartbeat failed')
