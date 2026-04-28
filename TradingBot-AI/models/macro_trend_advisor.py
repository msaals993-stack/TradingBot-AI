"""
📈 Macro Trend Advisor - Market Trend + Next Hour Prediction
Determines current market state and predicts next direction
"""

import sys


import time
import os
from typing import Optional

import numpy as np
import pandas as pd

# استيراد dl_client إذا كان متوفراً
try:
    from dl_client_v2 import DeepLearningClientV2
    DL_CLIENT_AVAILABLE = True
except ImportError:
    DeepLearningClientV2 = None
    DL_CLIENT_AVAILABLE = False


class MacroTrendAdvisor:
    """
    Determines overall market state and predicts next direction
    """

    MACRO_SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']

    CACHE_DURATION      = 900  # 15 دقيقة - حالة السوق (إشارات 1h/4h بطيئة، تحديث أسرع يعطي ضجيج)
    PREDICTION_CACHE_1H = 900  # 15 دقيقة - بيانات 1h
    PREDICTION_CACHE_4H = 900  # 15 دقيقة - بيانات 4h

    RSI_PERIOD        = 14
    MOMENTUM_LOOKBACK = 10

    def __init__(self, exchange=None, dl_client=None):
        self.exchange = exchange

        # حالة السوق
        self._last_status     : str   = '⚪ NEUTRAL'
        self._last_check_time : float = 0.0

        # ⏳ فلتر الصبر (Stability Timer)
        self._is_currently_bull : bool = False
        self._macro_start_time  : Optional[float] = None

        # المايكرو يستخدم dl_client المشترك المحمل مسبقاً
        self.dl_client = dl_client

        # استخدام النماذج المحملة من cache RAM مباشرة ككائنات جاهزة
        # تأكيد الأسماء الصحيحة بناءً على ما يتم حقنه في الداتابيز
        models_cache = getattr(self.dl_client, '_models', {}) if self.dl_client else {}
        
        self.volume_predictor     = models_cache.get('volume_pred')
        self.sentiment_analyzer   = models_cache.get('sentiment')
        self.liquidity_analyzer   = models_cache.get('liquidity')
        self.smart_money_tracker  = models_cache.get('smart_money')
        self.crypto_news_analyzer = models_cache.get('crypto_news')

        # كاش predict_market المجمّع
        self._last_prediction      : dict  = {}
        self._last_prediction_time : float = 0.0

        # كاش منفصل لكل timeframe
        self._last_1h_prediction      : dict  = {}
        self._last_1h_prediction_time : float = 0.0
        self._last_4h_prediction      : dict  = {}
        self._last_4h_prediction_time : float = 0.0
        self._last_15m_prediction      : dict  = {}
        self._last_15m_prediction_time : float = 0.0

    # ─────────────────────────────────────────────
    # Main Interface
    # ─────────────────────────────────────────────

    def get_macro_status(self) -> str:
        """الحالة الحقيقية من إشارات 1h + 4h الفعلية - تتحدث كل 5 دقائق"""
        if not self.exchange:
            return '⚪ NEUTRAL'

        if time.time() - self._last_check_time < self.CACHE_DURATION:
            return self._last_status

        try:
            pred = self.predict_market()

            # استخدام combined المحسوب بدل جمع short + medium منفصلين
            combined_dir = pred.get('combined', {}).get('direction', 'NEUTRAL')
            combined_str = pred.get('combined', {}).get('strength',  'NEUTRAL')

            if   combined_dir == 'BULLISH' and combined_str == 'STRONG':  final_status = '🟢 BULL_MARKET'
            elif combined_dir == 'BEARISH' and combined_str == 'STRONG':  final_status = '🔴 BEAR_MARKET'
            elif combined_dir == 'BULLISH':                                final_status = '🟢 MILD_BULL'
            elif combined_dir == 'BEARISH' and combined_str == 'CAUTION': final_status = '🔴 MILD_BEAR'   # ✅ FIX 2: كان SIDEWAYS
            elif combined_dir == 'BEARISH':                                final_status = '🔴 MILD_BEAR'
            elif combined_dir == 'MIXED'   and combined_str == 'CAUTION': final_status = '⚪ SIDEWAYS'
            else:                                                           final_status = '⚪ SIDEWAYS'

            # للعرض فقط
            short_bull  = pred.get('short',  {}).get('bull_score', 0)
            short_bear  = pred.get('short',  {}).get('bear_score', 0)
            medium_bull = pred.get('medium', {}).get('bull_score', 0)
            medium_bear = pred.get('medium', {}).get('bear_score', 0)
            total_bull  = short_bull  + medium_bull
            total_bear  = short_bear  + medium_bear

            # تحقق الداتابيز: لو BULL، قارن بالحالة المحفوظة قبل ساعتين
            if 'BULL' in final_status:
                historical = self._get_historical_status(hours_ago=2)
                if historical and 'BEAR' in historical:
                    final_status = '⚪ SIDEWAYS'  # تعارض مع الحالة السابقة

            # 🛡️ تطبيق فلتر الصبر: التأكد من استقرار الصعود لمدة 30 دقيقة (أو 5 مع التحقق السريع)
            is_bull_signal = 'BULL' in final_status
            required_mins = 30
            stability_info = ""

            # ⚡ التحقق السريع باستخدام نماذج DL المحملة من DB: إذا كان أكثر من 3 نماذج تشير إلى BULL، قلل الانتظار
            quick_verified = False
            if is_bull_signal:
                quick_data = self._get_quick_verification_data()
                if quick_data:
                    try:
                        bull_votes = 0
                        total_models = 0

                        # volume_pred (النموذج محمل ككائن LightGBM/XGBoost جاهز)
                        if self.volume_predictor:
                            spike_proba = self.volume_predictor.predict(quick_data)
                            if spike_proba > 0.6: bull_votes += 1
                            total_models += 1

                        # sentiment
                        if self.sentiment_analyzer:
                            sent_score = self.sentiment_analyzer.predict(quick_data)
                            if sent_score > 0.5: bull_votes += 1
                            total_models += 1

                        # liquidity
                        if self.liquidity_analyzer:
                            liq_score = self.liquidity_analyzer.predict(quick_data)
                            if liq_score > 0.5: bull_votes += 1
                            total_models += 1

                        # smart_money
                        if self.smart_money_tracker:
                            sm_score = self.smart_money_tracker.predict(quick_data)
                            if sm_score > 0.5: bull_votes += 1
                            total_models += 1

                        # crypto_news
                        if self.crypto_news_analyzer:
                            news_score = self.crypto_news_analyzer.predict(quick_data)
                            if news_score > 0.5: bull_votes += 1
                            total_models += 1

                        # 2 كود ثابت إضافي
                        # multi_timeframe_analyzer
                        try:
                            from .multi_timeframe_analyzer import MultiTimeframeAnalyzer
                            mtf_analyzer = MultiTimeframeAnalyzer()
                            # تحقق من bottom detection (إشارة BULL)
                            bottom_conf = mtf_analyzer.analyze_bottom(quick_data).get('confidence', 0)
                            if bottom_conf > 60:  # قاع قوي = BULL
                                bull_votes += 1
                            total_models += 1
                        except Exception:
                            pass

                        # trend_early_detector
                        try:
                            from .trend_early_detector import TrendEarlyDetector
                            ted = TrendEarlyDetector()
                            ted_result = ted.detect(quick_data)
                            if ted_result.get('direction') == 'BULLISH' and ted_result.get('score', 0) > 40:
                                bull_votes += 1
                            total_models += 1
                        except Exception:
                            pass

                        if total_models >= 5 and bull_votes >= 5:  # أكثر من 70% من 7 متفقة على BULL
                            required_mins = 1  # تسريع إلى دقيقة واحدة
                            quick_verified = True
                    except Exception as e:
                        print(f"⚠️ Error in DL model verification: {e}")

            if is_bull_signal:
                if not self._is_currently_bull or self._macro_start_time is None:
                    # بداية رصد محاولة صعود جديدة
                    self._is_currently_bull = True
                    self._macro_start_time = time.time()
                
                elapsed_mins = (time.time() - self._macro_start_time) / 60
                if elapsed_mins < required_mins:
                    # لم يستقر بعد -> نحجب إشارة الشراء
                    verify_type = "Quick" if quick_verified else "Standard"
                    stability_info = f" (⏳ {int(elapsed_mins)}/{required_mins}m {verify_type})"
                    final_status = f"⚪ STABILIZING ({int(elapsed_mins)}/{required_mins}m {verify_type})"
                else:
                    stability_info = " (✅ Stable)"
            else:
                # أي تراجع أو تذبذب يصفر العداد فوراً
                self._is_currently_bull = False
                self._macro_start_time = None

            short_p  = pred.get('short',  {}).get('prediction', '?')
            medium_p = pred.get('medium', {}).get('prediction', '?')
            s1 = '🟢' if 'BULL' in str(short_p)  else ('🔴' if 'BEAR' in str(short_p)  else '⚪')
            s4 = '🟢' if 'BULL' in str(medium_p) else ('🔴' if 'BEAR' in str(medium_p) else '⚪')

            self._display_info = {
                'status'    : final_status + stability_info if "STABILIZING" not in final_status else final_status,
                '1h_icon'   : s1,
                '4h_icon'   : s4,
                '1h_details': pred.get('short',  {}).get('details', ''),
                '4h_details': pred.get('medium', {}).get('details', ''),
                'total_bull': total_bull,
                'total_bear': total_bear,
            }

            self._last_status     = final_status
            self._last_check_time = time.time()
            return final_status

        except Exception as e:
            print(f"⚠️ MacroTrendAdvisor get_macro_status error: {e}")
            return '⚪ NEUTRAL'

    def get_display_info(self) -> dict:
        """معلومات العرض للهيدر"""
        if hasattr(self, '_display_info'):
            return self._display_info
        return {
            'status'    : self._last_status,
            '1h_icon'   : '⚪',
            '4h_icon'   : '⚪',
            '1h_details': '',
            '4h_details': '',
            'total_bull': 0,
            'total_bear': 0,
        }

    def can_aim_high(self) -> bool:
        return 'BULL' in self.get_macro_status()

    def invalidate_cache(self) -> None:
        """إعادة ضبط كل الكاش"""
        self._last_check_time          = 0.0
        self._last_prediction_time     = 0.0
        self._last_1h_prediction_time  = 0.0
        self._last_4h_prediction_time  = 0.0
        self._last_15m_prediction_time = 0.0

    # ─────────────────────────────────────────────
    # Indicators
    # ─────────────────────────────────────────────

    def _calculate_rsi(self, prices: pd.Series) -> float:
        delta = prices.diff()
        gain  = delta.where(delta > 0, 0.0).rolling(self.RSI_PERIOD).mean()
        loss  = (-delta.where(delta < 0, 0.0)).rolling(self.RSI_PERIOD).mean()
        rs    = gain / loss.replace(0, np.nan)
        return float((100 - (100 / (1 + rs))).iloc[-1])

    def _calculate_momentum(self, prices: pd.Series, current_price: float) -> float:
        past_price = float(prices.iloc[-self.MOMENTUM_LOOKBACK])
        if past_price == 0:
            return 0.0
        return ((current_price - past_price) / past_price) * 100

    # ═════════════════════════════════════════════
    # 🔮 Prediction System
    # ═════════════════════════════════════════════

    def predict_market(self) -> dict:
        """تتحدث كل 5 دقائق"""
        if not self.exchange:
            return self._empty_prediction()

        if time.time() - self._last_prediction_time < self.CACHE_DURATION:
            return self._last_prediction

        try:
            fast     = self._predict_timeframe('15m', 48)   # فلتر سريع — آخر 12 ساعة
            short    = self._predict_timeframe('1h',  24)
            medium   = self._predict_timeframe('4h',  24)
            combined = self._combine_timeframes(fast, short, medium)

            result = {
                'fast'    : fast,
                'short'   : short,
                'medium'  : medium,
                'combined': combined,
            }

            self._last_prediction      = result
            self._last_prediction_time = time.time()
            return result

        except Exception as e:
            print(f"⚠️ predict_market error: {e}")
            return self._empty_prediction()

    def _predict_timeframe(self, timeframe: str, limit: int) -> dict:
        """كل timeframe له كاش مستقل"""
        if timeframe == '15m':
            cache_dur = 300  # 5 دقائق — فلتر سريع
            last_time = self._last_15m_prediction_time
            last_pred = self._last_15m_prediction
        elif timeframe == '1h':
            cache_dur = self.PREDICTION_CACHE_1H
            last_time = self._last_1h_prediction_time
            last_pred = self._last_1h_prediction
        else:
            cache_dur = self.PREDICTION_CACHE_4H
            last_time = self._last_4h_prediction_time
            last_pred = self._last_4h_prediction

        # رجع الكاش لو ما انتهى
        if time.time() - last_time < cache_dur and last_pred:
            return last_pred

        # جلب بيانات جديدة من API
        predictions = []
        for symbol in self.MACRO_SYMBOLS:
            try:
                result = self._predict_symbol_direction(symbol, timeframe, limit)
                if result:
                    predictions.append(result)
            except Exception as e:
                print(f"⚠️ Prediction error {symbol} {timeframe}: {e}")

        if not predictions:
            return {'prediction': 'NEUTRAL', 'confidence': 50, 'reason': '⚪ No signals'}

        result = self._combine_predictions(predictions, timeframe)

        # حفظ في الكاش الصحيح
        if timeframe == '15m':
            self._last_15m_prediction      = result
            self._last_15m_prediction_time = time.time()
        elif timeframe == '1h':
            self._last_1h_prediction      = result
            self._last_1h_prediction_time = time.time()
        else:
            self._last_4h_prediction      = result
            self._last_4h_prediction_time = time.time()

        return result

    def _predict_symbol_direction(self, symbol: str, timeframe: str, limit: int) -> Optional[dict]:
        ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df    = pd.DataFrame(ohlcv, columns=['ts', 'o', 'h', 'l', 'c', 'v'])

        if len(df) < 10:
            return None

        bull_signals = 0
        bear_signals = 0

        # 1. الاتجاه الحقيقي - آخر 10 شموع بدلاً من 3 (weight 4)
        last_10_change = (
            (float(df['c'].iloc[-1]) - float(df['c'].iloc[-11]))
            / float(df['c'].iloc[-11]) * 100
        )
        # شرط أقوى: لازم يكون الصعود/الهبوط واضح
        if   last_10_change >  1.5: bull_signals += 4  # صعود قوي
        elif last_10_change >  0.5: bull_signals += 2  # صعود خفيف
        elif last_10_change < -1.5: bear_signals += 4  # هبوط قوي
        elif last_10_change < -0.5: bear_signals += 2  # هبوط خفيف
        # لو التغيير بين -0.5% و +0.5% = سوق عرضي (لا نضيف نقاط)

        # 2. RSI - تعديل الحدود (weight 3)
        rsi = self._calculate_rsi(df['c'])
        if   rsi > 75: bear_signals += 3  # تشبع شرائي قوي
        elif rsi > 65: bear_signals += 1  # تشبع شرائي خفيف
        elif rsi > 55: bull_signals += 1  # زخم صاعد
        elif rsi < 30: bull_signals += 3  # تشبع بيعي قوي
        elif rsi < 40: bull_signals += 1  # تشبع بيعي خفيف
        elif rsi > 45 and rsi < 55: pass  # منطقة محايدة

        # 3. Volume - فلتر الصعود الوهمي (weight 3)
        avg_vol   = float(df['v'].iloc[-20:].mean())
        last_vol  = float(df['v'].iloc[-1])
        vol_ratio = last_vol / avg_vol if avg_vol > 0 else 1.0

        # ✅ الفلتر الأهم: صعود بحجم ضعيف = فخ!
        if vol_ratio < 0.8 and last_10_change > 0.5:
            bear_signals += 4  # صعود وهمي بحجم ضعيف = احتمال انعكاس
        elif vol_ratio > 2.0 and last_10_change > 0:
            bull_signals += 3  # صعود بحجم قوي = اتجاه حقيقي
        elif vol_ratio > 2.0 and last_10_change < 0:
            bear_signals += 3  # هبوط بحجم قوي = اتجاه حقيقي
        elif vol_ratio < 0.5:
            bear_signals += 1  # حجم ضعيف = ضعف السوق

        # 4. MACD - اتجاه الزخم (weight 2)
        macd         = df['c'].ewm(span=12).mean() - df['c'].ewm(span=26).mean()
        macd_signal  = macd.ewm(span=9).mean()
        macd_hist    = macd - macd_signal
        hist_current = float(macd_hist.iloc[-1])
        hist_prev    = float(macd_hist.iloc[-2])

        # استخدام الهستوجرام بدلاً من MACD المباشر (أدق)
        if   hist_current > 0 and hist_current > hist_prev: bull_signals += 2
        elif hist_current > 0:                               bull_signals += 1
        elif hist_current < 0 and hist_current < hist_prev: bear_signals += 2
        elif hist_current < 0:                               bear_signals += 1

        # 5. EMA Cross قصير المدى (weight 1)
        ema_9  = float(df['c'].ewm(span=9).mean().iloc[-1])
        ema_21 = float(df['c'].ewm(span=21).mean().iloc[-1])
        if ema_9 > ema_21: bull_signals += 1
        else:              bear_signals += 1

        # 5b. فلتر الاتجاه الحقيقي EMA50/200 (weight 3) — يمنع الشراء في سوق هابط
        # هذا الفلتر هو الأهم: لو السعر تحت EMA200 = سوق هابط بغض النظر عن الإشارات القصيرة
        if len(df) >= 200:
            ema_50  = float(df['c'].ewm(span=50,  adjust=False).mean().iloc[-1])
            ema_200 = float(df['c'].ewm(span=200, adjust=False).mean().iloc[-1])
            price   = float(df['c'].iloc[-1])

            if   price > ema_200 and ema_50 > ema_200: bull_signals += 3  # اتجاه صاعد حقيقي
            elif price < ema_200 and ema_50 < ema_200: bear_signals += 3  # اتجاه هابط حقيقي
            elif price < ema_200:                       bear_signals += 2  # تحت EMA200 = خطر
            elif ema_50 < ema_200:                      bear_signals += 1  # death cross وشيك

        # 6. Reversal candle - شموع الانعكاس (weight 3)
        c_last       = float(df['c'].iloc[-1])
        o_last       = float(df['o'].iloc[-1])
        h_last       = float(df['h'].iloc[-1])
        l_last       = float(df['l'].iloc[-1])
        candle_body  = abs(c_last - o_last)
        candle_range = h_last - l_last

        if candle_range > 0:
            body_ratio   = candle_body / candle_range
            upper_shadow = h_last - max(c_last, o_last)
            lower_shadow = min(c_last, o_last) - l_last

            # شمعة دوجي أو شمعة ذيل طويل = انعكاس محتمل
            if   upper_shadow > candle_body * 2.5 and body_ratio < 0.3: bear_signals += 3
            elif lower_shadow > candle_body * 2.5 and body_ratio < 0.3: bull_signals += 3
            elif upper_shadow > candle_body * 1.5: bear_signals += 1
            elif lower_shadow > candle_body * 1.5: bull_signals += 1

        return {'symbol': symbol, 'bull': bull_signals, 'bear': bear_signals}

    # ─────────────────────────────────────────────
    # 🔮 Combine Predictions
    # ─────────────────────────────────────────────

    def _combine_predictions(self, predictions: list, timeframe: str) -> dict:
        total_bull = sum(p['bull'] for p in predictions)
        total_bear = sum(p['bear'] for p in predictions)
        total      = total_bull + total_bear

        if total == 0:
            return {'prediction': 'NEUTRAL', 'confidence': 50, 'reason': '⚪ No signals'}

        # ✅ شرط أقوى: لازم عملتين على الأقل + فرق واضح بين النقاط
        bull_count = sum(1 for p in predictions if p['bull'] > p['bear'] + 2)  # فرق نقطتين على الأقل
        bear_count = sum(1 for p in predictions if p['bear'] > p['bull'] + 2)
        if bull_count < 2 and bear_count < 2:
            details = ', '.join(
                f"{p['symbol'].split('/')[0]}(🟢{p['bull']}/🔴{p['bear']})"
                for p in predictions
            )
            return {
                'prediction': 'NEUTRAL',
                'confidence': 50,
                'bull_score': total_bull,
                'bear_score': total_bear,
                'reason'    : f'⚪ No consensus (2/3 required) - {"Next 1h" if timeframe == "1h" else "Next 4h"}',
                'details'   : details,
            }

        bull_pct = total_bull / total * 100
        bear_pct = total_bear / total * 100
        tf_label = 'Next 1h' if timeframe == '1h' else 'Next 4h'

        if   bull_pct >= 75: prediction, confidence, reason = 'STRONG_BULLISH', bull_pct, f'🟢🟢 Strong Bullish - {tf_label}'
        elif bull_pct >= 60: prediction, confidence, reason = 'BULLISH',        bull_pct, f'🟢 Bullish - {tf_label}'
        elif bear_pct >= 75: prediction, confidence, reason = 'STRONG_BEARISH', bear_pct, f'🔴🔴 Strong Bearish - {tf_label}'
        elif bear_pct >= 60: prediction, confidence, reason = 'BEARISH',        bear_pct, f'🔴 Bearish - {tf_label}'
        else:                prediction, confidence, reason = 'NEUTRAL',        50,       f'⚪ Neutral - {tf_label}'

        details = ', '.join(
            f"{p['symbol'].split('/')[0]}(🟢{p['bull']}/🔴{p['bear']})"
            for p in predictions
        )

        return {
            'prediction': prediction,
            'confidence': round(confidence, 1),
            'bull_score': total_bull,
            'bear_score': total_bear,
            'reason'    : reason,
            'details'   : details,
        }

    def _combine_timeframes(self, fast: dict, short: dict, medium: dict) -> dict:
        fast_pred   = fast.get('prediction',   'NEUTRAL')
        short_pred  = short.get('prediction',  'NEUTRAL')
        medium_pred = medium.get('prediction', 'NEUTRAL')

        is_fast_bear   = 'BEAR' in fast_pred
        is_fast_bull   = 'BULL' in fast_pred
        is_short_bull  = 'BULL' in short_pred
        is_short_bear  = 'BEAR' in short_pred
        is_medium_bull = 'BULL' in medium_pred
        is_medium_bear = 'BEAR' in medium_pred

        # ❌ فلتر الاتجاه: لو 15m هابط + 4h هابط = لا تشتري مهما كان 1h
        if is_fast_bear and is_medium_bear:
            return {'direction': 'BEARISH', 'strength': 'STRONG',
                    'confidence': max(fast['confidence'], medium['confidence']),
                    'reason': '🔴🔴 Bear Filter Active (15m + 4h Bearish — NO BUY)'}

        if   is_short_bull and is_medium_bull and is_fast_bull:
            return {'direction': 'BULLISH', 'strength': 'STRONG',
                    'confidence': max(short['confidence'], medium['confidence']),
                    'reason': '🟢🟢🟢 Confirmed Bullish (15m + 1h + 4h)'}
        elif is_short_bull and is_medium_bull:
            return {'direction': 'BULLISH', 'strength': 'STRONG',
                    'confidence': max(short['confidence'], medium['confidence']),
                    'reason': '🟢🟢 Confirmed Bullish (1h + 4h)'}
        elif is_short_bear and is_medium_bear:
            return {'direction': 'BEARISH', 'strength': 'STRONG',
                    'confidence': max(short['confidence'], medium['confidence']),
                    'reason': '🔴🔴 Confirmed Bearish (1h + 4h)'}
        elif is_short_bull and is_medium_bear:
            # ✅ FIX 1: 4h الهابط يتفوق على 1h الصاعد — الارتداد مؤقت فقط
            return {'direction': 'BEARISH', 'strength': 'CAUTION',  'confidence': 55,
                    'reason': '⚠️ 1h Bullish but 4h Bearish → 4h wins (Dead-cat bounce risk)'}
        elif is_short_bear and is_medium_bull:
            return {'direction': 'MIXED',   'strength': 'RECOVERY', 'confidence': 55,
                    'reason': '⏳ Temp Dip then Recovery (Wait)'}
        else:
            return {'direction': 'NEUTRAL', 'strength': 'NEUTRAL',  'confidence': 50,
                    'reason': '⚪ Market Unclear'}

    # ─────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────

    def _get_quick_verification_data(self) -> Optional[dict]:
        """جمع البيانات للتحقق السريع"""
        try:
            if not self.exchange:
                return None

            # جلب بيانات 1h لـ BTC/USDT
            ohlcv = self.exchange.fetch_ohlcv('BTC/USDT', '1h', limit=50)
            df = pd.DataFrame(ohlcv, columns=['ts', 'o', 'h', 'l', 'c', 'v'])

            if len(df) < 20:
                return None

            # حساب volume_ratio
            avg_vol = float(df['v'].iloc[-20:].mean())
            last_vol = float(df['v'].iloc[-1])
            volume_ratio = last_vol / avg_vol if avg_vol > 0 else 1.0

            # حساب price_change_1h (آخر ساعة)
            price_change_1h = ((float(df['c'].iloc[-1]) - float(df['c'].iloc[-2])) / float(df['c'].iloc[-2])) * 100

            # حساب momentum_strength (استخدام RSI كمقياس للزخم)
            rsi = self._calculate_rsi(df['c'])
            momentum_strength = rsi - 50  # مركزي حول 50

            # حساب bullish_volume و bearish_volume
            bullish_volume = 0
            bearish_volume = 0
            for i in range(-10, 0):  # آخر 10 شموع
                if float(df['c'].iloc[i]) > float(df['o'].iloc[i]):
                    bullish_volume += float(df['v'].iloc[i])
                else:
                    bearish_volume += float(df['v'].iloc[i])

            return {
                'volume_ratio': volume_ratio,
                'price_change_1h': price_change_1h,
                'momentum_strength': momentum_strength,
                'bullish_volume': bullish_volume,
                'bearish_volume': bearish_volume,
                'rsi': rsi,
                'macd_hist': 0,  # يمكن إضافة MACD إذا لزم الأمر
            }

        except Exception as e:
            print(f"⚠️ Quick verification data error: {e}")
            return None

    def _get_historical_status(self, hours_ago: float = 2) -> Optional[str]:
        """
        يقرأ macro_status المحفوظة في bot_settings (key=bot_status)
        ويرجع القيمة المحفوظة — تُستخدم للمقارنة قبل إرجاع BULL.
        الربط: advisor.db = your_database_storage_instance

        ✅ FIX 3: يستخدم 'timestamp' (unix) بدل 'time' (string) لتفادي
                  خطأ عبور منتصف الليل حين يصبح الفرق سالباً.
                  لو الداتابيز تحفظ 'time' فقط، نحسب التوافق بطريقة آمنة.
        """
        try:
            if not hasattr(self, 'db') or self.db is None:
                return None

            raw = self.db.load_setting('bot_status')
            if not raw:
                return None

            import json
            data = json.loads(raw)

            macro = data.get('macro_status', '')
            if not macro:
                return None

            max_age_secs = hours_ago * 3600 + 1800  # hours_ago + 30 دقيقة هامش

            # ── الطريقة الأولى: timestamp unix (الأفضل والأدق) ──────────────
            saved_ts = data.get('timestamp')
            if saved_ts:
                try:
                    diff_secs = time.time() - float(saved_ts)
                    if diff_secs < 0 or diff_secs > max_age_secs:
                        return None
                    return macro
                except Exception:
                    pass  # نكمل للطريقة الثانية

            # ── الطريقة الثانية: وقت النص "%H:%M:%S" (fallback آمن) ─────────
            # ✅ نعالج حالة عبور منتصف الليل بأخذ القيمة المطلقة أو إضافة يوم
            saved_time = data.get('time', '')
            if saved_time:
                try:
                    from datetime import datetime, timedelta
                    now      = datetime.now()
                    saved_dt = datetime.strptime(saved_time, "%H:%M:%S").replace(
                        year=now.year, month=now.month, day=now.day
                    )
                    diff_secs = (now - saved_dt).total_seconds()

                    # لو الفرق سالب = تجاوزنا منتصف الليل → نضيف يوم للسجل
                    if diff_secs < 0:
                        saved_dt  += timedelta(days=1)
                        diff_secs  = (now - saved_dt).total_seconds()

                    if diff_secs < 0 or diff_secs > max_age_secs:
                        return None
                except Exception:
                    pass  # لو فشل تحليل الوقت نكمل بالقيمة كما هي

            return macro

        except Exception:
            return None

    def _empty_prediction(self) -> dict:
        return {
            'fast'    : {'prediction': 'NEUTRAL', 'confidence': 50, 'reason': '⚪ No data'},
            'short'   : {'prediction': 'NEUTRAL', 'confidence': 50, 'reason': '⚪ No data'},
            'medium'  : {'prediction': 'NEUTRAL', 'confidence': 50, 'reason': '⚪ No data'},
            'combined': {'direction': 'NEUTRAL', 'strength': 'NEUTRAL', 'confidence': 50, 'reason': '⚪ No data'},
        }
