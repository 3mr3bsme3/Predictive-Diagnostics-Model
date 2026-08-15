import sys, os, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline import OBDIIDeploymentPipeline, IQRClipper, to_str_func
import __main__
__main__.OBDIIDeploymentPipeline = OBDIIDeploymentPipeline
__main__.IQRClipper = IQRClipper
__main__.to_str_func = to_str_func

import joblib, pandas as pd, numpy as np

model = joblib.load('best_model.pkl')

DATASET_DIR = r'e:\AAST\Term 7\Predictive Diagnostics Model\OBD-II datasets'
csv_files = sorted([f for f in os.listdir(DATASET_DIR) if f.endswith('.csv')])
dfs = [pd.read_csv(os.path.join(DATASET_DIR, f), low_memory=False) for f in csv_files]
raw_df = pd.concat(dfs, axis=0, ignore_index=True)

def parse_num(s):
    s = str(s).strip()
    for u in ['RPM', '%', 'kPa', 'C', 'km/h', 'km']:
        s = s.replace(u, '')
    s = s.replace(',', '.')
    try: return float(s)
    except: return np.nan

for col in raw_df.columns:
    if col not in ['AUTOMATIC', 'FUEL_TYPE', 'DTC_NUMBER', 'TROUBLE_CODES', 'MARK', 'MODEL', 'VEHICLE_ID']:
        raw_df[col] = raw_df[col].apply(parse_num)

raw_df['target'] = raw_df['DTC_NUMBER'].apply(lambda v: np.nan if pd.isna(v) else (0 if str(v).strip() == 'MIL is OFF0 codes' else 1))
raw_df.dropna(subset=['target'], inplace=True)
raw_df['target'] = raw_df['target'].astype(int)

s1 = raw_df[(raw_df['target']==0) & (raw_df['CAR_YEAR']>=2014)].iloc[10]
s2 = raw_df[(raw_df['target']==0) & (raw_df['SPEED'] < 40) & (raw_df['CAR_YEAR']==2011)].iloc[5]
s3 = raw_df[(raw_df['target']==0) & (raw_df['CAR_YEAR']==2012) & (raw_df['ENGINE_POWER']==1.6)].iloc[20]
s4 = raw_df[(raw_df['target']==1) & (raw_df['CAR_YEAR']==2006)].iloc[50]
s5 = raw_df[(raw_df['target']==1) & (raw_df['CAR_YEAR']==2003)].iloc[0]

real_cars = [
    ('🚗 مثال ١: سيارة عائلية 2014 على سرعة طريق (سليمة تماماً)', s1),
    ('🚗 مثال ٢: سيارة 2011 في زحمة المدينة (سليمة تماماً)', s2),
    ('🚗 مثال ٣: سيارة 2012 بمحرك 1.6L وسرعة متوسطة (سليمة)', s3),
    ('⚠️ مثال ٤: سيارة حقيقية مسجلة كود عطل في الحساسات (Failure)', s4),
    ('⚠️ مثال ٥: سيارة قديمة 2003 مسجلة أعطال في المحرك (Failure)', s5),
]

print('=' * 75)
print('>>> REAL OBD-II TELEMETRY SAMPLES FROM ACTUAL VEHICLES <<<')
print('=' * 75)

for title, row in real_cars:
    df_row = pd.DataFrame([row])
    pred = model.predict(df_row)[0]
    prob = model.predict_proba(df_row)[0][1] * 100
    dtc = str(row['DTC_NUMBER'])
    actual = 'سليمة - Normal (0)' if row['target'] == 0 else f'معطلة - Failure (1) [كود العطل: {dtc}]'
    
    print('\n' + title)
    print(f"  • سنة الصنع: {int(row['CAR_YEAR']) if pd.notna(row['CAR_YEAR']) else 'غير محدد'} | سعة الماتور: {row['ENGINE_POWER']}L | نوع الوقود: {row['FUEL_TYPE']}")
    print(f"  • حرارة التبريد: {row['ENGINE_COOLANT_TEMP']} °C | دوسة البنزين: {row['THROTTLE_POS']} % | السرعة: {row['SPEED']} km/h | RPM: {row['ENGINE_RPM']}")
    print(f"  • الواقع الفعلي المسجل في كمبيوتر السيارة: {actual}")
    print(f"  • تشخيص الذكاء الاصطناعي (الموديل) : {'🚨 عطل محتمل!' if pred==1 else '✅ سليمة تماماً'} (نسبة الخطر: {prob:.1f}%)")
    print('-' * 75)
