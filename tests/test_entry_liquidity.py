from datetime import datetime, timedelta, timezone
from types import SimpleNamespace as N
from unittest.mock import Mock
import pandas as pd
import pytest
from ccexchange.config import Liquidity, RuntimeSettings, load_config
from ccexchange.liquidity import entry_liquidity
from ccexchange import runtime
from ccexchange.execution import AccountSnapshot
from ccexchange.regime import MarketRegime
from ccexchange.state import StateStore


def quote(**kw):
    return dict(price=99.9,ask=100,ask_size=10,bid_size=10,
                timestamp=datetime.now(timezone.utc).isoformat(),**kw)


def test_liquidity_caps_to_ten_percent_of_displayed_ask():
    result=entry_liquidity(quote(),datetime.now(timezone.utc),Liquidity())
    assert result.max_notional==100


@pytest.mark.parametrize('field,value',[
    ('price',float('nan')),('ask',99),('ask',105),('ask_size',None),('ask_size',0),
    ('bid_size',None),('ask_size',float('inf')),('ask_size',.001),
    ('timestamp','2020-01-01T00:00:00Z'),('timestamp','2026-10-05T00:00:00'),
    ('timestamp',(datetime.now(timezone.utc)+timedelta(minutes=5)).isoformat())])
def test_bad_stale_wide_and_insufficient_quotes_fail_closed(field,value):
    q=quote();q[field]=value
    with pytest.raises((ValueError,TypeError)):
        entry_liquidity(q,datetime.now(timezone.utc),Liquidity())


def setup_cycle(tmp_path,monkeypatch,live=False):
    now=datetime.now(timezone.utc)
    cfg=load_config('config/default.yaml').model_copy(update={'symbols':['BTC/USD'],'minimum_history_bars':2})
    frame=pd.DataFrame({k:[v,v,v] for k,v in dict(open=100,high=101,low=99,close=100,
        volume=0,dollar_volume=0,atr=2,adx=25,volume_ratio=0,bb_width=.1).items()},
        index=pd.date_range(end=now-timedelta(hours=4),periods=3,freq='4h'))
    source=N(bars=lambda *a:{'BTC/USD':frame})
    broker=Mock()
    broker.account.return_value=AccountSnapshot(100000,100000,{})
    broker.filled_orders.return_value=[];broker.order_updates.return_value=[]
    broker.open_client_order_ids.return_value=set();broker.client=N(reconciliation_notes={})
    broker.quote.return_value=quote();broker.submit.return_value={'id':'paper-test'}
    monkeypatch.setattr(runtime,'load_config',lambda p:cfg)
    monkeypatch.setattr(runtime,'add_indicators',lambda frame,*a:frame)
    monkeypatch.setattr(runtime,'classify_regime',lambda *a:(MarketRegime.UPTREND,100))
    monkeypatch.setattr(runtime,'momentum_score',lambda *a:N(total=90,components={},reasons=['test']))
    monkeypatch.setattr(runtime,'required_entry_score',lambda *a:65)
    recorder=Mock();recorder.register_experiment.return_value='test-experiment'
    monkeypatch.setattr(runtime,'PaperRecorder',Mock(return_value=recorder))
    monkeypatch.setattr(runtime,'emit',Mock())
    settings=RuntimeSettings(_env_file=None,paper_trading=not live,live_trading=live,dry_run=False)
    audit=Mock();store=StateStore(tmp_path/'state.json')
    return broker,lambda:runtime.run_cycle(settings,audit,broker=broker,store=store,source=source,now=now),store


def test_paper_zero_bar_volume_can_enter_with_verified_quote_and_small_size(tmp_path,monkeypatch):
    broker,run,store=setup_cycle(tmp_path,monkeypatch)
    run()
    broker.submit.assert_called_once()
    assert broker.submit.call_args.args[0].quantity<=1
    assert broker.quote.call_count==2
    assert store.load().positions['BTC/USD'].entry_price==100
    run()
    broker.submit.assert_called_once()  # one decision per candle survives restart


def test_quote_is_rechecked_before_submission(tmp_path,monkeypatch):
    broker,run,_=setup_cycle(tmp_path,monkeypatch)
    wide=quote();wide['ask']=110
    broker.quote.side_effect=[quote(),wide]
    run();broker.submit.assert_not_called()


def test_live_trading_cannot_use_paper_quote_alternative(tmp_path,monkeypatch):
    broker,run,_=setup_cycle(tmp_path,monkeypatch,live=True)
    run();broker.submit.assert_not_called();broker.quote.assert_not_called()


def test_missing_quote_cannot_fall_back_to_looser_entry(tmp_path,monkeypatch):
    broker,run,_=setup_cycle(tmp_path,monkeypatch)
    broker.quote.side_effect=RuntimeError('data unavailable')
    run();broker.submit.assert_not_called()
