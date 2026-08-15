import os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline import OBDIIDeploymentPipeline, IQRClipper, to_str_func
import __main__
__main__.OBDIIDeploymentPipeline = OBDIIDeploymentPipeline
__main__.IQRClipper = IQRClipper
__main__.to_str_func = to_str_func

import joblib

# 1. تحميل الموديل المحفوظ
model = joblib.load('best_model.pkl')

# 2. بيانات عربية جاية من حساسات الـ OBD-II
car_data = {
    'CAR_YEAR': 2013,
    'THROTTLE_POS': 32.0,
    'ENGINE_COOLANT_TEMP': 95.0,
    'ENGINE_POWER': 1.6,
    'FUEL_TYPE': 'Biodiesel_Gasoline',
    'AUTOMATIC': 's',
    'ENGINE_RPM': 2200,
    'SPEED': 60.0
}

# 3. التنبؤ مباشرة (الموديل بينظف ويجهز البيانات ويتوقع تلقائياً)
prediction = model.predict(car_data)           # 0 = Normal, 1 = Failure
probability = model.predict_proba(car_data)    # نسبة الخطر [احتمال السليم, احتمال العطل]

print("=" * 50)
print("النتيجة:", "🚨 عطل محتمل!" if prediction[0] == 1 else "✅ العربية سليمة تماماً")
print(f"احتمال العطل: {probability[0][1] * 100:.1f}%")
print("=" * 50)
