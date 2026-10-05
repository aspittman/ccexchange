import contextlib
import fcntl
import json
import re
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

PREFIXES = {
    'ccexchange': ('ccexchange-',),
    'ETFEnhancer': ('etfenhancer-paper-', 'etfenhancer-live-'),
    'ETFEnhancerLT': ('etfenhancerlt-',),
    'momentum_master': ('momentum_master-',),
    'options_covered': ('covered_call_', 'covered_stock_'),
    'options_secured': ('cash_secured_put_', 'os-'),
    'options_inverted': ('long_put_', 'oi-'),
    'options_direct': ('long_call_',),
}
TERMINAL = {'filled', 'canceled', 'expired', 'rejected', 'replaced'}

class OwnershipError(RuntimeError):
    pass

def value(obj, name, default=None):
    return obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)

def enum(v):
    return str(getattr(v, 'value', v))

def instrument(symbol):
    # Slash is only a display difference for Alpaca crypto pairs. Preserve OCC
    # expiration, call/put and strike: different contracts remain independent.
    s = str(symbol).upper().replace('/', '')
    if not re.fullmatch(r'[A-Z0-9.]+', s):
        raise OwnershipError('Invalid instrument identity')
    return s

class GuardedClient:
    def __init__(self, client, trader_id, directory=None):
        if trader_id not in PREFIXES:
            raise OwnershipError('Unknown trader ID')
        self._client, self.trader_id = client, trader_id
        self.directory = Path(directory or Path(__file__).resolve().parent)
        self._identity = None

    @property
    def identity(self):
        if self._identity is None:
            identity = str(value(self._client.get_account(), 'id'))
            seed = json.loads((self.directory/'baseline.json').read_text())
            if identity not in seed['accounts'] or self.trader_id not in seed['accounts'][identity]['bots']:
                raise OwnershipError('Account is not registered for this trader')
            self._identity = identity
        return self._identity

    @contextlib.contextmanager
    def _locked(self):
        identity = self.identity
        # One lock across bots covers checks, reservation and broker response.
        # A committed pending reservation survives process/network failure.
        with (self.directory/'ownership.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            db = sqlite3.connect(self.directory/'ownership.sqlite3', timeout=30)
            db.row_factory = sqlite3.Row
            try:
                db.execute('CREATE TABLE IF NOT EXISTS claims(account TEXT, symbol TEXT, owner TEXT, pending TEXT, PRIMARY KEY(account,symbol))')
                db.execute('CREATE TABLE IF NOT EXISTS verified_orders(account TEXT, broker_id TEXT, owner TEXT, PRIMARY KEY(account,broker_id))')
                db.execute('CREATE TABLE IF NOT EXISTS requests(account TEXT, client_id TEXT, owner TEXT, symbol TEXT, broker_id TEXT, PRIMARY KEY(account,client_id))')
                seed = json.loads((self.directory/'baseline.json').read_text())['accounts'][identity]
                for symbol, row in seed['claims'].items():
                    db.execute('INSERT OR IGNORE INTO claims VALUES(?,?,?,NULL)', (identity,symbol,row['owner']))
                db.commit()
                yield db
            finally:
                db.close()
                fcntl.flock(lock, fcntl.LOCK_UN)

    def owns_order(self, order):
        cid = str(value(order, 'client_order_id', ''))
        if cid.startswith(PREFIXES[self.trader_id]):
            return True
        baseline = json.loads((self.directory/'baseline.json').read_text())['accounts'][self.identity]
        return baseline.get('legacy_orders', {}).get(str(value(order, 'id', ''))) == self.trader_id

    def _owner(self, db, symbol):
        return db.execute('SELECT * FROM claims WHERE account=? AND symbol=?', (self.identity,instrument(symbol))).fetchone()

    def get_all_positions(self):
        with self._locked() as db:
            positions = self._client.get_all_positions()
            self._reconcile_claims(db,positions)
            return [p for p in positions if (self._owner(db, value(p,'symbol')) or {'owner':None})['owner']==self.trader_id]

    def get_open_position(self, symbol):
        with self._locked() as db:
            self._reconcile_claims(db,self._client.get_all_positions())
            row = self._owner(db,symbol)
            if not row or row['owner']!=self.trader_id:
                # Existing bots recognize Alpaca's 404 as 'no owned position'.
                from alpaca.common.exceptions import APIError
                from requests import HTTPError, Response
                response = Response(); response.status_code = 404
                raise APIError('{"code":40410000,"message":"No position owned by this trader"}', HTTPError(response=response))
            return self._client.get_open_position(symbol)

    def get_orders(self, *args, **kwargs):
        return [o for o in self._client.get_orders(*args, **kwargs) if self.owns_order(o)]

    def get_order_by_id(self, order_id, *args, **kwargs):
        order = self._client.get_order_by_id(order_id, *args, **kwargs)
        if not self.owns_order(order):
            raise OwnershipError('Order belongs to another or unknown trader')
        return order

    def get_order_by_client_id(self, client_id, *args, **kwargs):
        order = self._client.get_order_by_client_id(client_id, *args, **kwargs)
        if not self.owns_order(order):
            raise OwnershipError('Order belongs to another or unknown trader')
        return order

    def _activities(self, kind, after):
        rows=[]; token=None
        for _ in range(100):
            params={'after':after,'direction':'asc','page_size':100}
            if token: params['page_token']=token
            page=self._client.get('/account/activities/'+kind, data=params)
            if not isinstance(page,list):
                raise OwnershipError('Invalid activity response')
            rows.extend(page)
            if len(page)<100: return rows
            nxt=page[-1]['id']
            if nxt==token: raise OwnershipError('Incomplete activity pagination')
            token=nxt
        raise OwnershipError('Activity history limit reached; reconciliation required')

    def _reconcile_claims(self, db, positions):
        # Also covers trades made by the old processes before the next cron
        # restart. Native prefixes are evidence; an untagged order is not.
        seed=json.loads((self.directory/'baseline.json').read_text())['accounts'][self.identity]
        if not positions: return
        after=seed['observed_at']
        expected={(symbol,row['owner']):float(row.get('quantity',0)) for symbol,row in seed['claims'].items()}
        unknown=set(); owners={}
        def stamp(text):
            text=re.sub(r'\.(\d+)',lambda m:'.'+m[1][:6].ljust(6,'0'),text)
            return datetime.fromisoformat(text.replace('Z','+00:00'))
        for fill in self._activities('FILL',after):
            if stamp(fill['transaction_time'])<=stamp(after): continue
            symbol=instrument(fill['symbol']); oid=fill['order_id']
            cached=db.execute('SELECT owner FROM verified_orders WHERE account=? AND broker_id=?',(self.identity,oid)).fetchone()
            if cached: owner=cached['owner']
            else:
                order=self._client.get_order_by_id(oid)
                cid=str(value(order,'client_order_id',''))
                owner=next((bot for bot,prefixes in PREFIXES.items() if cid.startswith(prefixes)),None)
                owner=owner or seed.get('legacy_orders',{}).get(oid)
                if owner not in seed['bots']: owner=None
                db.execute('INSERT INTO verified_orders VALUES(?,?,?)',(self.identity,oid,owner))
            if not owner:
                unknown.add(symbol);continue
            owners.setdefault(symbol,set()).add(owner)
            pair=(symbol,owner)
            expected[pair]=expected.get(pair,0)+float(fill['qty'])*(1 if fill['side']=='buy' else -1)
        if any(enum(value(p,'asset_class'))=='crypto' for p in positions):
            for fee in self._activities('CFEE',after):
                if not fee.get('symbol') or not fee.get('qty') or stamp(fee['created_at'])<=stamp(after):continue
                symbol=instrument(fee['symbol']); candidates=owners.get(symbol,set())
                if len(candidates)!=1:unknown.add(symbol);continue
                pair=(symbol,next(iter(candidates)))
                expected[pair]=expected.get(pair,0)+float(fee['qty'])
        for p in positions:
            symbol=instrument(value(p,'symbol')); qty=float(value(p,'qty'))
            balances=[(owner,q) for (s,owner),q in expected.items() if s==symbol and abs(q)>1e-8]
            if symbol in unknown or len(balances)!=1 or abs(balances[0][1]-qty)>1e-8:
                # Never retain a stale ownership claim when broker activity
                # contradicts it. The caller cannot manage uncertain holdings.
                if (self._owner(db,symbol) or {'owner':None})['owner']==self.trader_id:
                    raise OwnershipError('Position ownership no longer reconciles: '+symbol)
                continue
            owner=balances[0][0]; row=self._owner(db,symbol)
            if row and row['owner']!=owner:
                raise OwnershipError('Registry conflicts with broker fill ownership: '+symbol)
            db.execute('INSERT OR IGNORE INTO claims VALUES(?,?,?,NULL)',(self.identity,symbol,owner))
        db.commit()

    def _broker_state(self):
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus
        positions = self._client.get_all_positions()
        orders = self._client.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=500, nested=True))
        if len(orders)>=500:
            raise OwnershipError('Open-order response may be incomplete')
        flattened=[]
        def visit(order):
            flattened.append(order)
            for leg in value(order,'legs',None) or []: visit(leg)
        for order in orders: visit(order)
        return positions, flattened

    def _reserve(self, db, symbol, cid):
        key = instrument(symbol)
        positions, orders = self._broker_state()  # Errors propagate; never assume flat.
        self._reconcile_claims(db,positions)
        held = any(instrument(value(p,'symbol'))==key and float(value(p,'qty'))!=0 for p in positions)
        open_orders = [o for o in orders if instrument(value(o,'symbol'))==key]
        row = self._owner(db,key)
        if any(not self.owns_order(o) for o in open_orders):
            raise OwnershipError('Instrument has another trader or unknown open order')
        if row and row['pending']:
            # Even a 404 cannot prove a timed-out submission was never accepted.
            pending = self._client.get_order_by_client_id(row['pending'])
            if enum(value(pending,'status')) not in TERMINAL:
                if row['owner'] != self.trader_id:
                    raise OwnershipError('Instrument has an unresolved reservation')
            else:
                db.execute('UPDATE claims SET pending=NULL WHERE account=? AND symbol=?',(self.identity,key))
        if held and (not row or row['owner'] != self.trader_id):
            raise OwnershipError('Instrument position belongs to another or unknown trader')
        if row and row['owner']!=self.trader_id and open_orders:
            raise OwnershipError('Instrument is reserved by another trader')
        # Ownership can transfer only after broker-confirmed flat/no open orders
        # and successful resolution of the previous submission.
        db.execute('INSERT OR REPLACE INTO claims VALUES(?,?,?,?)',(self.identity,key,self.trader_id,cid))
        db.execute('INSERT INTO requests VALUES(?,?,?,?,NULL)',(self.identity,cid,self.trader_id,key))
        db.commit()

    def _release_rejected_stop(self, db, cid, exc):
        """Release only an explicit broker stop-price validation rejection.

        Timeouts, missing orders, duplicate IDs and all other ambiguous errors
        retain their reservation. Keep the request row as an audit trail.
        """
        from alpaca.common.exceptions import APIError
        if not isinstance(exc, APIError):
            return
        try:
            rejected = (exc.status_code == 422 and exc.code == 42210000
                        and exc.message == "stop price must be less than current price")
        except (ValueError, KeyError, TypeError, AttributeError):
            return
        if rejected:
            db.execute('UPDATE claims SET pending=NULL WHERE account=? AND owner=? AND pending=?',
                       (self.identity, self.trader_id, cid))
            db.commit()

    def submit_order(self, order_data=None, **kwargs):
        if order_data is None:
            raise OwnershipError('Explicit order request required')
        if value(order_data,'legs') or enum(value(order_data,'order_class','simple')) not in ('simple','None'):
            raise OwnershipError('Multi-leg/bracket requests require explicit ownership support')
        cid = value(order_data,'client_order_id')
        if not cid:
            cid = PREFIXES[self.trader_id][0]+uuid.uuid4().hex[:24]
            order_data.client_order_id = cid
        if not str(cid).startswith(PREFIXES[self.trader_id]) or len(cid)>48:
            raise OwnershipError('Client order ID does not identify this trader')
        symbol = value(order_data,'symbol')
        with self._locked() as db:
            self._reserve(db,symbol,cid)
            try:
                result = self._client.submit_order(order_data=order_data, **kwargs)
            except Exception as exc:
                self._release_rejected_stop(db, cid, exc)
                raise
            db.execute('UPDATE requests SET broker_id=? WHERE account=? AND client_id=?',(str(value(result,'id')),self.identity,cid))
            db.commit()
            return result

    def cancel_order_by_id(self, order_id, *args, **kwargs):
        with self._locked() as db:
            order = self.get_order_by_id(order_id)
            row = self._owner(db,value(order,'symbol'))
            if not row or row['owner'] != self.trader_id:
                raise OwnershipError('Cannot cancel an order on another trader instrument')
            return self._client.cancel_order_by_id(order_id,*args,**kwargs)

    def replace_order_by_id(self, order_id, order_data=None, **kwargs):
        with self._locked() as db:
            order = self.get_order_by_id(order_id)
            row = self._owner(db,value(order,'symbol'))
            if not row or row['owner']!=self.trader_id:
                raise OwnershipError('Cannot replace an order on another trader instrument')
            cid = value(order_data,'client_order_id') or PREFIXES[self.trader_id][0]+uuid.uuid4().hex[:24]
            if not cid.startswith(PREFIXES[self.trader_id]):
                raise OwnershipError('Replacement ID does not identify this trader')
            order_data.client_order_id = cid
            # Replacement stays on the owned instrument. Reserve before request.
            self._reserve(db,value(order,'symbol'),cid)
            try:
                result = self._client.replace_order_by_id(order_id,order_data=order_data,**kwargs)
            except Exception as exc:
                self._release_rejected_stop(db, cid, exc)
                raise
            db.execute('UPDATE requests SET broker_id=? WHERE account=? AND client_id=?',(str(value(result,'id')),self.identity,cid));db.commit()
            return result

    def __getattr__(self, name):
        # Explicit read-only forwarding. No generic POST/PATCH/DELETE or bulk
        # cancel/close/exercise route can bypass the guarded methods above.
        if name.startswith('_'):
            raise AttributeError(name)
        if name == 'get' or name.startswith('get_'):
            return getattr(self._client,name)
        raise OwnershipError('Unsupported trading client operation: '+name)
