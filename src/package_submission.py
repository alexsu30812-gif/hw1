"""Validate predictions, regenerate the report and assemble a clean hand-in archive."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
from html import escape
import json
from pathlib import Path
import shutil
import tempfile
import zipfile

from build_summary import build
from build_report import build_report, validate_payload

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def validate_predictions(root, predictions):
    audit = read(root / 'artifacts/data_audit.json')
    expected = read(root / 'results/training_audit.json')
    scores = read(root / 'results/test_scores.json')
    if set(predictions) != {'dataset_A', 'dataset_B'}:
        raise ValueError('Prediction dataset keys must match the handout')
    counts = {}
    for short in ['A', 'B']:
        key = f'dataset_{short}'
        rows = predictions[key]
        wanted = expected['datasets'][short]['test_ids']
        if set(rows) != set(wanted) or len(rows) != audit[key]['counts']['test']:
            raise ValueError(f'{key}: incomplete or extra test IDs')
        labels = audit[key]['labels']
        for sample_id, top3 in rows.items():
            if not isinstance(top3, list) or len(top3) != 3 or len(set(top3)) != 3:
                raise ValueError(f'{sample_id}: need three distinct labels')
            if any(label not in labels for label in top3) or sample_id.endswith('.wav'):
                raise ValueError(f'{sample_id}: invalid label or ID')
        for item in scores['datasets'][short]['predictions']:
            rank = sorted(range(6), key=lambda i: (-item['scores'][i], i))
            if rows[item['sample_id']] != [labels[i] for i in rank[:3]]:
                raise ValueError('Top-3 order differs from saved classifier confidence')
        counts[key] = len(rows)
    reproduced = root / 'results/reproduced_predictions.json'
    if not reproduced.exists() or predictions != read(reproduced):
        raise ValueError('Fresh CPU inference must reproduce every submitted ranking')
    return counts


def guide(summary, final):
    supervised, alm_rows = [], []
    for key, dataset in summary['datasets'].items():
        selected = next(r for r in dataset['results'] if r['name'] == dataset['selected_model'])
        supervised.append(f'<tr><td>{escape(key)}</td><td>{escape(selected["name"])}</td>'
                          f'<td>{selected["top1"]:.2%}</td><td>{selected["top3"]:.2%}</td></tr>')
        a = summary['alm']['results'][key][0]
        alm_rows.append(f'<tr><td>{key}</td><td>{a["n_samples"]}</td><td>{a["top1"]:.2%}</td><td>{a["top3"]:.2%}</td></tr>')
    status = ('報告已填入你提供的公開連結。上傳前請確認該連結可由未登入的瀏覽器開啟，且包含完整程式與模型。'
              if final else '目前唯一未完成的提交欄位是「公開程式與模型資料夾連結」。PDF 已標記 DRAFT；補上連結後才是可提交的正式版。')
    if 'github.com' in summary['student'].get('cloud_url', ''):
        status += ' 本版依你提供的網址使用 GitHub；作業原文指定 cloud-drive folder，若助教限定雲端硬碟，需把同一份檔案鏡像至 Drive 並更新報告連結。'
    return '''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>HW1 作業與方法說明 — r14942154</title><style>
body{font-family:system-ui,"PingFang TC",sans-serif;line-height:1.8;color:#183047;background:#f3f6fa;margin:0}main{max-width:920px;margin:auto;padding:42px 24px}section{background:white;padding:24px 30px;margin:20px 0;border-radius:12px}h1,h2,h3{line-height:1.35}h2{color:#086b81}table{border-collapse:collapse;width:100%;font-size:.95rem}th,td{padding:10px;text-align:left;border-bottom:1px solid #dce3ec}code{background:#eef3f7;padding:3px 6px;border-radius:4px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#eef3f7;padding:16px}a{color:#066b97}.notice{border-left:5px solid #d29922;padding-left:20px}.flow{background:#e8f3f5;padding:16px;border-radius:8px}small{color:#506375}@media print{body{background:white}section{break-inside:avoid;padding:12px 0}main{padding:0}}
</style><main><h1>HW1：做了什麼、為什麼這樣做</h1><p>學號 r14942154 · 正式報告使用英文 · 本頁供你理解方法與操作提交</p>
<section class="notice"><h2>提交狀態</h2><p>''' + status + '''</p>
<p>prediction 已由實際模型產生，並要求 CPU 從音檔重新推論的全部 234 筆 Top-3 與提交檔完全一致，打包程式才會通過。</p></section>
<section><h2>1. 作業到底要完成什麼？</h2>
<p>這是兩個六分類問題。Dataset A 判斷美國發行音樂的年代：1960s 到 2010s；Dataset B 判斷 1980 年代音樂的發行市場：US、UK、Brazil、Spain、Germany、Italy。<strong>市場不等於歌手國籍或歌曲語言。</strong></p>
<p>A 的 train／validation／test 是 1026／132／132 筆；B 是 798／102／102 筆。每筆是 30 秒、mono、24 kHz 音訊。train 用於學習；validation 有標籤，可計算結果；test 沒有標籤，只能產生答案，不能宣稱 test 正確率。</p>
<p>每筆 test 要輸出三個不同類別，依信心由高到低排列。Top-1 是第一名正確的比例，Top-3 是正解落在前三名的比例。混淆矩陣的橫列為真實類別、直欄為預測類別；對角線是答對筆數。</p>
<p>另須用音訊語言模型（ALM）跑兩個資料集，完整覆蓋所選 split，交代 prompt、答案解析、錯誤處理並報告 Top-1、Top-3 和混淆矩陣。這一項不是第 14 頁列出的選做實驗。</p></section>
<section><h2>2. 使用 AI 有什麼限制？</h2>
<p>提供的 Homework_1.pdf 與 lecture01_intro.pdf 沒有明文訂出「生成式 AI 可協助到哪個程度」。這不等於老師已明確允許整份代寫。課程要求避免作弊與抄襲，使用公開程式、模型與論文時必須引用；你也表示沒有看到額外公告。</p>
<p>本份報告與 README 已揭露 Codex 協助寫程式、執行實驗、整理分析與英文撰稿。所有分數都来自真實運算，沒有捏造實驗，也沒有加入額外有標籤音訊。你應讀懂本頁與報告，尤其是模型選擇、資料切分、Top-3 與 ALM 的限制。</p></section>
<section><h2>3. 整體方法</h2><div class="flow">音訊 → 傳統特徵／MERT 特徵 → 標準化 → L2 正規化 → 分類器 → 六類機率 → Top-3 JSON</div>
<h3>傳統特徵：310 個數字描述音樂</h3><p>MFCC、log-mel 用來描述音色與頻譜能量；chroma 描述十二個音高類別的分布；RMS 描述能量；節奏特徵描述週期性變化。這些值本身不是年代答案，分類器必須從訓練集學習它們與類別的關係。</p>
<h3>MERT：用預訓練音樂模型取得特徵</h3><p>使用凍結的 MERT-v1-95M。把 30 秒分成六段固定 5 秒音訊，從第 7 與第 12 層取得每個時間點的 768 維表示。對所有時間點各算平均與標準差，得到 2 層 × 2 種統計 × 768 = <strong>3072 維</strong>。MERT 的權重沒有用本作業微調。</p>
<p>平均描述典型狀態，標準差描述變動程度；這稱為 pooling，可讓不同時間點的特徵變成固定長度向量。代價是失去事件先後順序。再與 310 維傳統特徵串接，得到 <strong>3382 維</strong>。</p>
<h3>標準化與正規化為什麼不同？</h3><p><code>z = (x − 訓練平均) / 訓練標準差</code> 讓各特徵的數值尺度可比較，避免單位較大的特徵占優勢。接著 <code>u = z / ||z||₂</code> 把每首歌的向量長度縮放至 1。平均與標準差只從當次訓練資料估計，不能先偷看 validation／test。</p>
<h3>分類器如何學習？</h3><p>比較 logistic regression 與 RBF SVC。前者學習各特徵的線性權重；後者透過相似度 <code>exp(−γ ||u−v||²)</code> 建立非線性的分類邊界。C 控制正則化強度，較大 C 通常較重視配合訓練資料。SVC 另以訓練資料做機率校準，最後按六類機率排序。</p></section>
<section><h2>4. 怎麼選模型才不會偷看答案？</h2>
<p>預先固定三種特徵路線（classical、MERT、concat）及七種分類器設定。只在官方 train 內做三折交叉驗證：每次拿其中兩折訓練，一折評估，讓所有 train 樣本都得到一次未參與該次訓練的預測。每一折重新估計 scaler。</p>
<p>用 <code>Top-1 + 0.5 × Top-3</code> 選定設定，再用完整官方 train 重訓。這只是選模型的指標，不是承諾最終成績的計分公式。正式 validation 僅拿來報告結果，不因結果好壞換模型；test 完全不參與選擇。官方切分保持不變。</p>
<p>官方 split 是 artist-disjoint；匿名資料沒有提供 artist ID，因此 train 內部三折無法再按歌手分組。這項限制已放進報告。</p>
<h3>真正量到的 validation 結果</h3><table><tr><th>資料集</th><th>提交的模型</th><th>Top-1</th><th>Top-3</th></tr>''' + ''.join(supervised) + '''</table>
<p>A 選中 concat + RBF SVC（C=10）；B 選中 concat + logistic regression（C=100）。兩者都由 train 內交叉驗證決定。報告另列三條路線的比較、各類別混淆、年代相鄰錯誤及具體錯例。這些結果表示音訊保留了一些可分類訊息，不能直接證明模型依據某個樂器或製作技巧做決策。</p></section>
<section><h2>5. ALM 實驗在做什麼？</h2><p>另外使用凍結的 TinyMU：音訊編碼器把音樂變成表示，投影層把它轉為語言模型可讀的向量，再搭配英文問題。依官方推論慣例，只用每段最前面的 10 秒並重採樣為 16 kHz。兩個完整 validation split 共 234 筆都已執行。</p>
<p>這次採「候選答案機率」方法：把六個合法類別分別當作候選回答，計算真實音訊和 prompt 條件下，每個答案 token（含結尾 EOS）的平均 log probability，再排序。較高代表模型在這個條件下較偏好該答案，<strong>不等於經校準的分類正確率</strong>。</p>
<p>由於只對合法候選排序，因此每筆都有合法且不重複的三個答案，invalid-format=0 是設計保證，不代表模型自由生成 JSON 的能力。非有限值或音訊讀取錯誤會停止運算，不能塞入假答案。只採一個預先固定的短 prompt 設計，符合模型短輸入與本機算力，沒有在 validation 上搜尋較好的 prompt。</p>
<table><tr><th>資料集</th><th>完整樣本數</th><th>Top-1</th><th>Top-3</th></tr>''' + ''.join(alm_rows) + '''</table>
<p>ALM 表現接近或低於均勻隨機排序的期望值（Top-1 16.7%、Top-3 50%）。B 全部 Top-1 都是 US，顯示明顯的答案偏好。原因可能包含 prompt、類別 tokenization 與預訓練資料；單一實驗無法分離原因。這個負面結果仍是有效的實驗結果，提交 JSON 使用上述訓練過的分類器。</p></section>
<section><h2>6. 要交哪些檔案？</h2><ol>
<li>把壓縮檔解開後的程式、模型、README 與 requirements 放入<strong>公開可存取</strong>的雲端資料夾。包內不含課程資料集、.venv 或預訓練模型快取。</li>
<li>把 <code>r14942154_report.pdf</code> 上傳到 NTU COOL 報告欄位，把 <code>r14942154.json</code> 上傳到 prediction 欄位。</li>
<li>同一個公開資料夾連結必須出現在<strong>報告第一頁</strong>及 NTU COOL <code>HW1_report</code> 的 <strong>comments</strong>。</li></ol>
<p>若尚未提供公開連結，可在 hw1 目錄執行以下命令補入連結、重建正式 PDF 與壓縮檔：</p>
<pre>.venv/bin/python src/package_submission.py --cloud-url 'https://你的公開資料夾連結'</pre>
<p>助教重現 prediction 的命令與固定套件版本已放在 README。完整驗證紀錄在 <code>results/submission_verification.json</code>；報告與程式保留來源引用及實驗紀錄。</p></section>
<section><h2>7. 需要記住的限制</h2><ul><li>validation 只是固定小樣本切分的測量，不能保證隱藏 test 的成績。</li><li>音樂年代／市場可能有重疊風格；市場尤其不是語言或國籍辨識。</li><li>預訓練模型是否曾接觸這些錄音無法完全排除；我們沒有額外蒐集有標籤音訊。</li><li>音訊由 Opus 解碼為 WAV，格式轉換無法恢復壓縮時已丟失的資訊。</li></ul></section>
<p><small>方法來源與完整超參數請見英文報告、README，以及可執行的 Python 程式。</small></p></main></html>'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cloud-url', help='Public folder containing the code and trained classifiers')
    args = p.parse_args()
    profile_path = ROOT / 'artifacts/student_profile.json'
    profile = read(profile_path)
    if args.cloud_url:
        from urllib.parse import urlparse
        u = urlparse(args.cloud_url)
        if u.scheme not in ['http', 'https'] or not u.netloc:
            p.error('--cloud-url must be a real HTTP(S) folder URL')
        profile['cloud_url'] = args.cloud_url
        write(profile_path, profile)
    sid = profile['student_id']
    if not sid.isalnum():
        raise ValueError('Student ID must be alphanumeric')
    predictions = read(ROOT / 'results/predictions.json')
    counts = validate_predictions(ROOT, predictions)
    write(ROOT / f'{sid}.json', predictions)
    summary = build(ROOT / 'results', ROOT / 'artifacts/data_audit.json', profile_path)
    write(ROOT / 'results/summary.json', summary)
    report = build_report(summary, ROOT / f'{sid}_report.pdf')
    final = not report['draft']
    (ROOT / 'HW1_中文說明.html').write_text(guide(summary, final), encoding='utf-8')
    verification = {
        'checked_at_utc': datetime.now(timezone.utc).isoformat(),
        'test_counts': counts, 'valid_distinct_top3': True,
        'descending_confidence_matches_saved_scores': True,
        'fresh_cpu_inference_all_rankings_exact_match': True,
        'cpu_inference_used_feature_cache': False,
        'report_pages': report['pages'], 'report_is_final': final,
        'remaining_report_fields': report['draft_reasons'],
        'public_folder_url': profile.get('cloud_url', ''),
        'public_folder_access_independently_verified': False,
        'alm_complete': True,
    }
    write(ROOT / 'results/submission_verification.json', verification)
    output = ROOT / 'submission'
    output.mkdir(exist_ok=True)
    name = f'{sid}_HW1'
    with tempfile.TemporaryDirectory(prefix='hw1-package-', dir=output) as temp:
        dest = Path(temp) / name
        dest.mkdir()
        files = [ROOT / n for n in ['.gitignore', 'README.md', 'requirements.txt', 'requirements-experiments.txt',
                 f'{sid}.json', f'{sid}_report.pdf', 'HW1_中文說明.html']]
        files += list((ROOT / 'src').glob('*.py'))
        files += [f for f in (ROOT / 'vendor/TinyMU').rglob('*') if f.is_file() and '__pycache__' not in f.parts]
        files += list((ROOT / 'artifacts/models').glob('*.joblib'))
        files += [ROOT / 'artifacts/data_audit.json', profile_path]
        files += list((ROOT / 'results').glob('*.json'))
        files += list((ROOT / 'results/alm').glob('*.json'))
        files += list((ROOT / 'results/alm').glob('*.jsonl'))
        hashes = {}
        for source in sorted(set(files)):
            relative = source.relative_to(ROOT)
            target = dest / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            hashes[str(relative)] = hashlib.sha256(target.read_bytes()).hexdigest()
        write(dest / 'FILE_SHA256.json', hashes)
        archive = ROOT / f'{name}.zip'
        with zipfile.ZipFile(archive.with_suffix('.zip.tmp'), 'w', zipfile.ZIP_DEFLATED) as z:
            for f in sorted(dest.rglob('*')):
                if f.is_file():
                    z.write(f, f.relative_to(dest.parent))
        archive.with_suffix('.zip.tmp').replace(archive)
        final_dir = output / name
        if final_dir.exists():
            # This directory is generated by this script and contains copies only.
            shutil.rmtree(final_dir)
        shutil.move(str(dest), final_dir)
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:
            raise ValueError('Archive integrity check failed')
        if any('/data/' in n or '/cache/' in n or '/.venv/' in n for n in z.namelist()):
            raise ValueError('An excluded dataset/cache file entered the archive')
    print(json.dumps({'archive': str(archive), 'bytes': archive.stat().st_size,
                      'verification': verification}, indent=2))


if __name__ == '__main__':
    main()
