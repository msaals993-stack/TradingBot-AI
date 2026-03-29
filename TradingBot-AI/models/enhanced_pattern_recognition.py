"""
🧠 Enhanced Pattern Recognition Model
يتعلم من الأنماط بشكل أعمق ويتوقع النجاح
"""

from datetime import datetime, timedelta
import statistics
from src.config import VOLUME_SPIKE_FACTOR

class EnhancedPatternRecognition:
    def __init__(self, storage):
        self.storage = storage
        self.all_patterns = []
        self.success_patterns = []
        self.trap_patterns = []
        self.load_patterns_from_db()
        print(f"🧠 Enhanced Pattern Recognition initialized with {len(self.all_patterns)} patterns cached.")

    def load_patterns_from_db(self):
        """Loads all patterns from the database and caches them."""
        self.all_patterns = self.storage.load_all_patterns()
        self.success_patterns = [p for p in self.all_patterns if p.get('pattern_type') == 'SUCCESS']
        self.trap_patterns = [p for p in self.all_patterns if p.get('pattern_type') == 'TRAP']

    
    def analyze_entry_pattern(self, symbol, analysis, mtf, price_drop):
        """تحليل نمط الدخول"""
        try:
            # استخراج الخصائص
            features = self._extract_features(analysis, mtf, price_drop)
            
            # البحث عن أنماط مشابهة
            similar_success = self._find_similar_patterns(features, 'SUCCESS')
            similar_traps = self._find_similar_patterns(features, 'TRAP')
            
            # حساب احتمال النجاح
            success_probability = self._calculate_success_probability(
                similar_success, similar_traps
            )
            
            # تحديد قوة النمط
            pattern_strength = self._calculate_pattern_strength(
                features, similar_success, similar_traps
            )
            
            # التوصية
            recommendation = self._get_pattern_recommendation(
                success_probability, pattern_strength
            )
            
            return {
                'success_probability': round(success_probability, 2),
                'pattern_strength': pattern_strength,
                'recommendation': recommendation,
                'similar_success_count': len(similar_success),
                'similar_trap_count': len(similar_traps),
                'confidence_adjustment': self._calculate_confidence_adjustment(
                    success_probability, pattern_strength
                ),
                'features': features
            }
            
        except Exception as e:
            print(f"⚠️ Pattern stats error: {e}")
            return None

    def analyze_peak_hunter_pattern(self, candles):
        """Analyzes the last 10 candles to identify robust reversal patterns."""
        if not candles or len(candles) < 10:
            return {'signal': 'neutral', 'reason': 'Not enough data (requires 10 candles)'}

        # Last 2 candles for pattern detection, previous 8 for trend context
        trend_candles = candles[:8]
        pattern_candles = candles[8:]

        prev_candle = pattern_candles[0]
        curr_candle = pattern_candles[1]

        # --- Data Validation ---
        required_keys = ['open', 'close', 'high', 'low', 'volume']
        if not all(key in prev_candle and key in curr_candle for key in required_keys):
            return {'signal': 'neutral', 'reason': 'Incomplete candle data'}

        # --- 1. Micro-Trend Analysis (first 8 candles) ---
        trend_start_price = trend_candles[0]['open']
        trend_end_price = trend_candles[-1]['close']
        is_micro_uptrend = trend_end_price > trend_start_price
        is_micro_downtrend = trend_end_price < trend_start_price

        # --- 2. Candlestick Pattern Analysis (last 2 candles) ---
        body_size = abs(curr_candle['close'] - curr_candle['open'])
        candle_range = curr_candle['high'] - curr_candle['low']
        if candle_range == 0: # Avoid division by zero for doji candles
            return {'signal': 'neutral', 'reason': 'Doji candle, indecision'}
            
        upper_wick = curr_candle['high'] - max(curr_candle['open'], curr_candle['close'])
        lower_wick = min(curr_candle['open'], curr_candle['close']) - curr_candle['low']

        # --- 3. Signal Confirmation (Pattern + Trend + Volume) ---
        is_volume_high = curr_candle['volume'] > prev_candle['volume'] * VOLUME_SPIKE_FACTOR

        # Hammer Confirmation: Must appear after a micro-downtrend.
        is_hammer = lower_wick > body_size * 2 and upper_wick < body_size * 0.5
        if is_hammer and is_micro_downtrend and is_volume_high:
            return {'signal': 'buy', 'reason': f'Hammer confirmed after downtrend with high volume'}

        # Shooting Star Confirmation: Must appear after a micro-uptrend.
        is_shooting_star = upper_wick > body_size * 2 and lower_wick < body_size * 0.5
        if is_shooting_star and is_micro_uptrend and is_volume_high:
            return {'signal': 'sell', 'reason': f'Shooting Star confirmed after uptrend with high volume'}

        # Original Peak Hunter Logic (Green -> Red with Volume) - now requires a micro-uptrend
        is_prev_green = prev_candle['close'] > prev_candle['open']
        is_curr_red = curr_candle['close'] < curr_candle['open']
        if is_prev_green and is_curr_red and is_micro_uptrend and is_volume_high:
            return {'signal': 'sell', 'reason': 'Peak Hunter confirmed after uptrend with high volume'}

        return {'signal': 'neutral', 'reason': 'No confirmed pattern'}
    
    def _extract_features(self, analysis, mtf, price_drop):
        """استخراج الخصائص من البيانات"""
        import pandas as pd
        
        features = {}
        
        # التأكد من وجود البيانات
        if not analysis or not isinstance(analysis, dict):
            analysis = {}
        
        # RSI features
        rsi = analysis.get('rsi', 50)
        if not pd.isna(rsi):
            features['rsi'] = rsi
            features['rsi_zone'] = self._get_rsi_zone(rsi)
        
        # MACD features
        macd_diff = analysis.get('macd_diff', 0)
        if not pd.isna(macd_diff):
            features['macd_diff'] = macd_diff
            features['macd_signal'] = 'bullish' if macd_diff > 0 else 'bearish'
        
        # Volume features
        volume_ratio = analysis.get('volume_ratio', 1.0)
        if not pd.isna(volume_ratio):
            features['volume_ratio'] = volume_ratio
            features['volume_level'] = self._get_volume_level(volume_ratio)
        
        # Momentum features
        momentum = analysis.get('price_momentum', 0)
        if not pd.isna(momentum):
            features['momentum'] = momentum
            features['momentum_direction'] = 'up' if momentum > 0 else 'down'
        
        # Trend features
        if mtf and isinstance(mtf, dict):
            features['trend'] = mtf.get('trend', 'neutral')
            features['trend_strength'] = mtf.get('total', 0)
        else:
            features['trend'] = 'neutral'
            features['trend_strength'] = 0
        
        # Price drop features
        if price_drop and isinstance(price_drop, dict):
            features['price_drop'] = price_drop.get('drop_percent', 0)
            features['drop_confirmed'] = price_drop.get('confirmed', False)
        else:
            features['price_drop'] = 0
            features['drop_confirmed'] = False
        
        return features
    
    def _get_rsi_zone(self, rsi):
        """تحديد منطقة RSI"""
        if rsi < 25:
            return 'extreme_oversold'
        elif rsi < 30:
            return 'oversold'
        elif rsi < 40:
            return 'low'
        elif rsi < 60:
            return 'neutral'
        elif rsi < 70:
            return 'high'
        elif rsi < 75:
            return 'overbought'
        else:
            return 'extreme_overbought'
    
    def _get_volume_level(self, volume_ratio):
        """تحديد مستوى Volume"""
        if volume_ratio < 0.5:
            return 'very_low'
        elif volume_ratio < 0.8:
            return 'low'
        elif volume_ratio < 1.2:
            return 'normal'
        elif volume_ratio < 2.0:
            return 'high'
        elif volume_ratio < 3.0:
            return 'very_high'
        else:
            return 'extreme'
    
    def _find_similar_patterns(self, features, pattern_type):
        """Finds similar patterns by searching the in-memory cache."""
        # 1. Select the correct cached list based on pattern_type
        candidate_patterns = []
        if pattern_type == 'SUCCESS':
            candidate_patterns = self.success_patterns
        elif pattern_type == 'TRAP':
            candidate_patterns = self.trap_patterns
        else:
            return [] # Should not happen

        # 2. Perform detailed similarity calculation in memory on the cached subset
        similar = []
        for pattern in candidate_patterns:
            # The data structure from the cache is already pattern['data']['features']
            pattern_features = pattern.get('data', {}).get('features', {})
            
            if not pattern_features:
                continue
            
            # Filter by key features before doing expensive calculations
            if features.get('rsi_zone') != pattern_features.get('rsi_zone'):
                continue
            if features.get('trend') != pattern_features.get('trend'):
                continue

            similarity = self._calculate_similarity(features, pattern_features)
            
            if similarity > 0.65:  # 65% threshold
                similar.append({
                    'pattern': pattern,
                    'similarity': similarity
                })
        
        # 3. Sort by similarity and return the top 10
        similar.sort(key=lambda x: x['similarity'], reverse=True)
        
        return similar[:10]
    
    def _calculate_similarity(self, features1, features2):
        """حساب التشابه بين نمطين"""
        import pandas as pd
        
        score = 0
        count = 0
        
        # RSI zone
        if 'rsi_zone' in features1 and 'rsi_zone' in features2:
            if features1['rsi_zone'] == features2['rsi_zone']:
                score += 1
            count += 1
        
        # MACD signal
        if 'macd_signal' in features1 and 'macd_signal' in features2:
            if features1['macd_signal'] == features2['macd_signal']:
                score += 1
            count += 1
        
        # Volume level
        if 'volume_level' in features1 and 'volume_level' in features2:
            if features1['volume_level'] == features2['volume_level']:
                score += 1
            count += 1
        
        # Trend
        if 'trend' in features1 and 'trend' in features2:
            if features1['trend'] == features2['trend']:
                score += 1
            count += 1
        
        # Momentum direction
        if 'momentum_direction' in features1 and 'momentum_direction' in features2:
            if features1['momentum_direction'] == features2['momentum_direction']:
                score += 1
            count += 1
        
        # RSI numerical
        if 'rsi' in features1 and 'rsi' in features2:
            rsi1 = features1['rsi']
            rsi2 = features2['rsi']
            if not pd.isna(rsi1) and not pd.isna(rsi2):
                diff = abs(rsi1 - rsi2)
                score += max(0, 1 - diff / 100)
                count += 1
        
        # Volume ratio
        if 'volume_ratio' in features1 and 'volume_ratio' in features2:
            vol1 = features1['volume_ratio']
            vol2 = features2['volume_ratio']
            if not pd.isna(vol1) and not pd.isna(vol2):
                diff = abs(vol1 - vol2)
                score += max(0, 1 - diff / 3)
                count += 1
        
        return score / count if count > 0 else 0
    
    def _calculate_success_probability(self, similar_success, similar_traps):
        """حساب احتمال النجاح"""
        if not similar_success and not similar_traps:
            return 0.5  # 50% افتراضي
        
        success_weight = sum(s['similarity'] for s in similar_success)
        trap_weight = sum(t['similarity'] for t in similar_traps)
        
        total_weight = success_weight + trap_weight
        
        if total_weight == 0:
            return 0.5
        
        probability = success_weight / total_weight
        
        return probability
    
    def _calculate_pattern_strength(self, features, similar_success, similar_traps):
        """حساب قوة النمط"""
        strength = 'WEAK'
        
        total_similar = len(similar_success) + len(similar_traps)
        
        if total_similar >= 10:
            strength = 'STRONG'
        elif total_similar >= 5:
            strength = 'MEDIUM'
        
        # تعديل حسب التشابه
        if similar_success:
            avg_similarity = statistics.mean([s['similarity'] for s in similar_success])
            if avg_similarity > 0.85:
                if strength == 'MEDIUM':
                    strength = 'STRONG'
                elif strength == 'WEAK':
                    strength = 'MEDIUM'
        
        return strength
    
    def _get_pattern_recommendation(self, success_probability, pattern_strength):
        """الحصول على التوصية"""
        if success_probability >= 0.75 and pattern_strength == 'STRONG':
            return 'STRONG_BUY'
        elif success_probability >= 0.65 and pattern_strength in ['STRONG', 'MEDIUM']:
            return 'BUY'
        elif success_probability >= 0.50:  # Changed from 0.55
            return 'CONSIDER'
        elif success_probability >= 0.40:  # Changed from 0.45
            return 'NEUTRAL'
        elif success_probability >= 0.30:  # Changed from 0.35 - more aggressive
            return 'CAUTION'
        else:
            return 'AVOID'
    
    def _calculate_confidence_adjustment(self, success_probability, pattern_strength):
        """حساب تعديل الثقة"""
        adjustment = 0
        
        # حسب الاحتمال
        if success_probability >= 0.80:
            adjustment += 10
        elif success_probability >= 0.70:
            adjustment += 7
        elif success_probability >= 0.60:
            adjustment += 5
        elif success_probability >= 0.50:
            adjustment += 2
        elif success_probability < 0.40:
            adjustment -= 5
        elif success_probability < 0.30:
            adjustment -= 10
        
        # حسب القوة
        if pattern_strength == 'STRONG':
            adjustment += 3
        elif pattern_strength == 'MEDIUM':
            adjustment += 1
        
        return adjustment
