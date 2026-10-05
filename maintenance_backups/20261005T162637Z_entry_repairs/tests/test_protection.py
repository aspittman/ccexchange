from datetime import datetime, timezone, timedelta
from types import SimpleNamespace
import json
import threading
from pathlib import Path
import pytest
from ccexchange.config import load_config
from ccexchange.execution import AccountSnapshot, BrokerPosition
from ccexchange.state import BotState, ManagedPosition, StateStore
from ccexchange.protection import protect_cycle, fresh_quote
from ccexchange.runtime import run_cycle

class Audit:
    def __init__(self):self.events=[]
    def write(self,event,**values):self.events.append((event,values))

class Broker:
    def __init__(self):
        self.qty=1.;self.price=105.;self.orders=[];self.status=None;self.timeout=False;self.quote_age=0
    def account(self):
        positions={'BTC/USD':BrokerPosition('BTC/USD',self.qty,100,self.qty*self.price,self.price)} if self.qty else {}
        return AccountSnapshot(10000,9000,positions)
    def quote(self,symbol):return {'price':self.price,'ask':self.price+1,'timestamp':(datetime.now(timezone.utc)-timedelta(seconds=self.quote_age)).isoformat()}
    def lookup_order(self,cid):return self.status
    def submit(self,intent):
        self.orders.append(intent)
        if self.timeout:raise TimeoutError('ambiguous POST')
        return {'id':'broker-'+str(len(self.orders))}

@pytest.fixture
def setup(tmp_path,monkeypatch):
    cfg=load_config('config/default.yaml');monkeypatch.chdir(tmp_path)
    store=StateStore(tmp_path/'state/bot_state.json')
    store.save(BotState(positions={'BTC/USD':ManagedPosition('BTC/USD',100,1,95,100,datetime.now(timezone.utc).isoformat(),atr=2,trade_id='entry-test',entry_order_id='entry-id')}))
    return SimpleNamespace(settings=SimpleNamespace(dry_run=False),cfg=cfg,store=store,broker=Broker(),audit=Audit())

def tick(s,now=None):protect_cycle(s.settings,s.cfg,s.audit,s.broker,s.store,now)

def test_ratchet_persists_restart_and_emits_linked_changes(setup):
    s=setup;tick(s);assert s.store.load().positions['BTC/USD'].stop==100
    s.store=StateStore(s.store.path);s.broker.price=109;tick(s)
    assert s.store.load().positions['BTC/USD'].stop==104
    s.broker.price=106;tick(s);assert s.store.load().positions['BTC/USD'].stop==104
    stops=[e for e in map(json.loads,open('logs/monitor_events.jsonl')) if e['event_type']=='STOP_UPDATED']
    assert len(stops)==2
    assert all(e['trade_id']=='entry-test' and e['order_id']=='entry-id' and e['timestamp'] for e in stops)

def test_outage_does_not_duplicate_exit_even_after_restart(setup):
    s=setup;s.broker.price=90;s.broker.timeout=True;tick(s)
    cid=s.store.load().exit_orders['BTC/USD']['client_id'];assert len(s.broker.orders)==1
    s.store=StateStore(s.store.path);tick(s)
    assert len(s.broker.orders)==1
    assert s.store.load().exit_orders['BTC/USD']['client_id']==cid
    assert 'BTC/USD' in s.store.load().protection_issues

def test_rejected_order_retries_only_after_terminal_confirmation(setup):
    s=setup;s.broker.price=90;tick(s);now=datetime.now(timezone.utc)
    s.broker.status={'id':'broker-1','status':'rejected','filled_qty':0};tick(s,now)
    assert len(s.broker.orders)==1
    tick(s,now+timedelta(seconds=61));assert len(s.broker.orders)==2
    assert s.broker.orders[0].client_order_id!=s.broker.orders[1].client_order_id

def test_partial_fill_then_cancellation_retries_only_remaining_balance(setup):
    s=setup;s.broker.price=90;tick(s)
    s.broker.qty=.4;s.broker.status={'id':'broker-1','status':'partially_filled','filled_qty':.6}
    tick(s);assert len(s.broker.orders)==1
    now=datetime.now(timezone.utc);s.broker.status['status']='canceled';tick(s,now)
    s.broker.price=110;tick(s,now+timedelta(seconds=61))
    assert s.broker.orders[-1].quantity==.4 and len(s.broker.orders)==2

def test_filled_exit_and_flat_position_close_without_resubmission(setup):
    s=setup;s.broker.price=90;tick(s)
    s.broker.qty=0;s.broker.status={'id':'broker-1','status':'filled','filled_qty':1};tick(s)
    assert not s.store.load().positions and not s.store.load().exit_orders
    tick(s);assert len(s.broker.orders)==1

def test_stale_quotes_preserve_stops_and_mark_degraded(setup):
    s=setup;s.broker.quote_age=180;s.broker.price=90;tick(s)
    assert not s.broker.orders and s.store.load().positions['BTC/USD'].stop==95
    assert 'BTC/USD' in json.loads(s.store.path.with_name('protection_health.json').read_text())['issues']

def test_telemetry_failure_does_not_block_protection(setup,monkeypatch):
    s=setup;s.broker.price=90;original=Path.open
    def failing(path,*args,**kwargs):
        if path.name=='monitor_events.jsonl':raise OSError('disk unavailable')
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',failing);tick(s);assert len(s.broker.orders)==1

def test_state_failure_prevents_submission(setup,monkeypatch):
    s=setup;s.broker.price=90
    monkeypatch.setattr(s.store,'save',lambda _: (_ for _ in ()).throw(OSError('state disk failure')))
    with pytest.raises(OSError):tick(s)
    assert not s.broker.orders

def test_indicator_download_failure_cannot_block_stop_check(setup):
    s=setup;s.broker.price=90;blocked=threading.Event();release=threading.Event();errors=[]
    class Source:
        def bars(self,*args):
            blocked.set();release.wait(2);raise RuntimeError('indicator download failed')
    settings=SimpleNamespace(config_path=str(Path(__file__).parents[1]/'config/default.yaml'),dry_run=False)
    def scan():
        try:run_cycle(settings,s.audit,broker=s.broker,store=s.store,source=Source())
        except RuntimeError as exc:errors.append(str(exc))
    worker=threading.Thread(target=scan);worker.start();assert blocked.wait(1)
    tick(s);release.set();worker.join()
    assert len(s.broker.orders)==1 and errors==['indicator download failed']

def test_unknown_legacy_pending_sell_is_not_resent(setup):
    s=setup;state=s.store.load();state.pending_orders['BTC/USD']='sell';s.store.save(state)
    s.broker.price=90;tick(s)
    assert not s.broker.orders
    assert 'Legacy pending' in s.store.load().protection_issues['BTC/USD']

def test_quote_rejects_nan_future_and_crossed_prices():
    now=datetime.now(timezone.utc)
    for price,ask,stamp in ((float('nan'),1,now),(2,1,now),(1,2,now+timedelta(seconds=90))):
        with pytest.raises(ValueError):fresh_quote({'price':price,'ask':ask,'timestamp':stamp.isoformat()},now)

def test_pending_fee_inventory_uses_actual_quantity_and_reports_provisional(setup):
    s=setup;s.broker.qty=.9975;s.broker.price=90
    s.broker.client=SimpleNamespace(reconciliation_notes={'BTCUSD':{'status':'FEE_RECONCILIATION_PENDING'}})
    tick(s)
    assert s.broker.orders[0].quantity==.9975
    health=json.loads(s.store.path.with_name('protection_health.json').read_text())
    assert health['reconciliation']['BTCUSD']['status']=='FEE_RECONCILIATION_PENDING'

def test_watchdog_restarts_stalled_cycle_and_allows_progress(monkeypatch):
    import ast, inspect, os
    import ccexchange.runtime as runtime
    tree=ast.parse(inspect.getsource(runtime))
    definition=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='watchdog')
    module=ast.Module(body=[definition],type_ignores=[]);ast.fix_missing_locations(module)
    class Exit(Exception):pass
    exits=[]
    def leave(code):exits.append(code);raise Exit()
    monkeypatch.setattr(os,'_exit',leave);monkeypatch.setattr(os,'write',lambda *a:None)
    namespace={'time':SimpleNamespace(monotonic=lambda:121),'progress':[0],
               'scan_stop':SimpleNamespace(wait=lambda seconds:False)}
    exec(compile(module,'watchdog-test','exec'),namespace)
    with pytest.raises(Exit):namespace['watchdog']()
    assert exits==[1]
    waits=iter([False,True]);namespace['progress'][0]=120
    namespace['scan_stop']=SimpleNamespace(wait=lambda seconds:next(waits))
    namespace['watchdog']();assert exits==[1]
