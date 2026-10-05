"""Convert measured training and ALM outputs into the report's validated schema."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from build_report import validate_payload


def read(path):
    return json.loads(Path(path).read_text())


def build(results_dir, data_audit, profile):
    cv = read(results_dir / 'cv_results.json')
    val = read(results_dir / 'validation_results.json')
    audit = read(results_dir / 'training_audit.json')
    alm = read(results_dir / 'alm' / 'alm.json')
    data = read(data_audit)
    if not alm['complete'] or alm['n_completed'] != alm['n_expected']:
        raise ValueError('Audio language model evaluation is incomplete')
    if audit['official_validation_used_for_selection_or_fitting'] or audit['test_labels_used']:
        raise ValueError('Training protocol does not meet the documented split policy')
    selected_parameters = {}
    datasets = {}
    for short in ['A', 'B']:
        key = f'dataset_{short}'
        vc, cc = val['datasets'][short], cv['datasets'][short]
        route = cc['selected_route']
        best = max(cc['routes'], key=lambda k: cc['routes'][k]['selected_cv_score'])
        if route != best or vc['selected_route'] != route:
            raise ValueError('The reported selected model is not the train-CV winner')
        selected = vc['routes'][route]
        winner = next(c for c in cc['routes'][route]['candidates']
                      if c['candidate_id'] == cc['selected_candidate_id'])
        selected_parameters[short] = f"{route}: {winner['candidate_id']}"
        results = []
        for name, item in vc['routes'].items():
            m = item['metrics']
            results.append({'name': f"{name} / {item['candidate_id']}",
                            'top1': m['top1_accuracy'], 'top3': m['top3_accuracy'],
                            'n_samples': m['sample_count'],
                            'confusion_matrix': m['confusion_matrix_counts']})
        metrics = selected['metrics']
        labels, cm = vc['classes'], metrics['confusion_matrix_counts']
        pairs = sorted([(cm[i][j], labels[i], labels[j]) for i in range(6)
                        for j in range(6) if i != j], reverse=True)
        analysis = [f"Selected using train-only CV: {route}/{selected['candidate_id']}; "
                    f"OOF Top-1 + 0.5 Top-3 = {selected['cv_selection_score']:.4f}."]
        analysis += [f"Most frequent directed error: {truth} to {prediction} ({count} recordings)."
                     for count, truth, prediction in pairs[:2] if count]
        if short == 'A':
            adjacent = metrics['adjacent_decade_error_count']
            errors = metrics['error_count']
            analysis.append(f"Adjacent-decade mistakes account for {adjacent}/{errors} Top-1 errors "
                            f"({100 * adjacent / errors if errors else 0:.1f}%).")
            neighbor_pairs = sorted([(cm[i][i+1] + cm[i+1][i], labels[i], labels[i+1])
                                     for i in range(5)], reverse=True)
            analysis.append('Largest neighboring confusions (both directions): ' + '; '.join(
                f'{left}/{right}: {count}' for count,left,right in neighbor_pairs[:2]) + '.')
            analysis.append('Production, timbre and instrumentation can provide decade cues; '
                            'the confusion matrix alone does not establish which cues caused a decision.')
        else:
            analysis.append('Release market is not nationality or language. Shared musical styles '
                            'are a possible explanation for confusion, not an established causal finding.')
        learned = vc['routes']['mert']['metrics']['top1_accuracy']
        traditional = vc['routes']['classical']['metrics']['top1_accuracy']
        combined = vc['routes']['concat']['metrics']['top1_accuracy']
        comparison = (
            f"After separate train-CV selection, MERT changes Top-1 by {100*(learned-traditional):+.1f} "
            f"percentage points versus classical; concatenation changes it by {100*(combined-learned):+.1f} versus MERT. "
            "This compares full pipelines, not an isolated causal effect of a feature."
        )
        best_class = max(range(6), key=lambda i: cm[i][i] / sum(cm[i]))
        learned_evidence = (
            f"The selected pipeline most reliably identifies {labels[best_class]} "
            f"({cm[best_class][best_class]}/{sum(cm[best_class])} correct). "
            "This demonstrates separable audio-label patterns for that class in this split, "
            "but does not identify individual instruments or production techniques."
        )
        datasets[key] = {'labels': labels, 'counts': data[key]['counts'],
                         'selected_model': f"{route} / {selected['candidate_id']}",
                         'results': results,
                         'errors': [{'sample_id': e['sample_id'], 'true': e['true_label'],
                                     'predicted': e['top1'], 'top3': e['top3']}
                                    for e in selected['errors']], 'analysis': analysis,
                         'representation_comparison': comparison, 'learned_evidence': learned_evidence}
    cfg = alm['config']
    prompt_id = 'fixed_closed_choice'
    alm_results = {}
    alm_notes = []
    for key in datasets:
        a = alm['datasets'][key]
        m = a['metrics_all_samples']
        if not a['complete'] or a['n_completed'] != data[key]['counts']['validation']:
            raise ValueError('ALM must cover the complete validation split for both tasks')
        alm_results[key] = [{'prompt_id': prompt_id, 'top1': m['top1_accuracy'],
                             'top3': m['top3_accuracy'], 'n_samples': m['n_labelled'],
                             'confusion_matrix': m['confusion_matrix_counts'],
                             'invalid_outputs': a['first_response_invalid_count'],
                             'retries': a['generation_attempt_count'] - a['n_completed'],
                             'fallbacks': a['fallback_count']}]
        totals = [sum(row[j] for row in m['confusion_matrix_counts']) for j in range(6)]
        most = max(range(6), key=lambda j: totals[j])
        alm_notes.append(f"{key}: most frequent Top-1 answer was {m['labels'][most]} "
                         f"({totals[most]}/{m['n_labelled']} clips).")
    alm_notes.extend([
        'Uniform random distinct rankings have expected Top-1 16.7% and Top-3 50.0%. This run shows weak task transfer.',
        'Class preference may reflect the prompt, answer tokenization or pretraining. One prompt cannot identify the cause.',
        'Zero invalid formats is guaranteed by scoring a fixed candidate set; it is not a free-generation result.',
    ])
    versions = {k: audit['packages'][k] for k in ['numpy','scipy','scikit-learn']}
    versions.update({'torch': '2.8.0', 'transformers': '4.57.1', 'librosa': '0.11.0'})
    references = [
        {'title':'Li et al. MERT: Acoustic Music Understanding with Large-Scale Self-supervised Training',
         'url':'https://arxiv.org/abs/2306.00107','used_for':'Frozen MERT-v1-95M audio representation.'},
        {'title':'MERT-v1-95M official model and implementation',
         'url':'https://huggingface.co/m-a-p/MERT-v1-95M','used_for':'Pinned pretrained weights and encoder code; revision 12af15f.'},
        {'title':'TinyMU: official music audio-language model source',
         'url':'https://github.com/xiquan-li/TinyMU','used_for':'MATPAC++ encoder and projector source; commit 385133d.'},
        {'title':'Li, Quelennec and Essid. TinyMU: A Compact Audio-Language Model for Music Understanding',
         'url':'https://arxiv.org/abs/2604.15849','used_for':'Architecture and design of the pretrained ALM comparison.'},
        {'title':'TinyMU official pretrained checkpoint',
         'url':'https://huggingface.co/AndreasXi/TinyMU','used_for':'Full trained model; revision 0735fc5. No fine-tuning on course data.'},
        {'title':'SmolLM2-135M tokenizer and architecture configuration',
         'url':'https://huggingface.co/HuggingFaceTB/SmolLM2-135M','used_for':'TinyMU language decoder configuration/tokenizer; trained weights restored from TinyMU.'},
        {'title':'librosa 0.11.0', 'url':'https://librosa.org/doc/0.11.0/',
         'used_for':'MFCC, log-mel, spectral, chroma and rhythm features; resampling.'},
        {'title':'Scikit-learn', 'url':'https://scikit-learn.org/stable/',
         'used_for':'Fold-local scaling, L2 normalization, logistic regression, calibrated SVC and CV.'},
        {'title':'PyTorch and TorchAudio', 'url':'https://pytorch.org/',
         'used_for':'Frozen audio encoder and audio-language-model execution.'},
        {'title':'Hugging Face Transformers', 'url':'https://github.com/huggingface/transformers',
         'used_for':'MERT/HuBERT components and TinyMU SmolLM2 decoder/tokenizer.'},
        {'title':'PyTorch Image Models (timm)', 'url':'https://github.com/huggingface/pytorch-image-models',
         'used_for':'Transformer components used by the upstream MATPAC++ implementation.'},
        {'title':'ReportLab', 'url':'https://www.reportlab.com/',
         'used_for':'Programmatic PDF layout, tables and confusion-matrix graphics.'},
    ]
    payload = {'student': read(profile), 'datasets': datasets,
        'experiment': {
            'seed': 42, 'device': 'Apple Silicon MPS; CPU feature processing and classifiers',
            'feature_method': 'Classical, MERT, or concatenated features',
            'feature_details': [
                'Classical: 310 dimensions; MFCC and deltas, 64-band log-mel, spectral, chroma, energy and rhythm statistics.',
                'MERT-v1-95M: frozen 94.4M-parameter encoder. Six fixed non-overlapping 5-second chunks cover each 30-second clip.',
                'Hidden layers 7 and 12: mean and population standard deviation over all chunk frames; 3,072 dimensions.',
                'Concatenation: 3,382 dimensions. All three feature routes use the same train-only selection procedure.',
            ],
            'preprocessing': [
                'Use supplied mono 24 kHz PCM16 WAV clips and unchanged official splits; verify audio SHA256.',
                'Classical STFT: 2,048-sample FFT, 512-sample hop, 64 mel bands; no per-recording gain normalization.',
                'MERT uses its official waveform processor and fixed chunk positions; no encoder fine-tuning or augmentation.',
                'StandardScaler fits only the current training fold; L2 normalization then scales each feature vector.',
            ],
            'classifier': 'Logistic regression or RBF SVC, independently selected for each task',
            'hyperparameters': {'Selected A': selected_parameters['A'], 'Selected B': selected_parameters['B'],
                                'LR C': '0.1, 1, 10, 100', 'SVC C': '1, 10, 100',
                                'SVC gamma': 'scale; probability=True', 'CV': '3 stratified folds; seed 42'},
            'training_protocol': [
                'Use only the official train split for 3-fold stratified cross-validation.',
                'Each scaler and classifier is fitted inside its training fold. Predeclare all routes and hyperparameters.',
                'Select route and classifier by pooled OOF Top-1 + 0.5 Top-3; deterministic first-candidate tie rule.',
                'Refit each route winner on the complete official train split. Evaluate official validation once for reporting.',
                'The final selected route remains the train-CV winner, regardless of validation rankings. Test labels are hidden.',
            ],
            'selection_rule': 'Maximize train-only OOF Top-1 + 0.5 Top-3; no official validation/test tuning',
            'commands': ['pip install -r requirements.txt',
                         'python src/infer.py --data-root /path/to/data --model-dir artifacts/models --output r14942154.json --device cpu'],
            'artifacts': ['A_selected.joblib; B_selected.joblib', 'Pinned feature extractors and label order',
                          'README and dependency files', 'Prediction JSON and submission-format validator'],
            'versions': versions,
            'limitations': [
                'Small, balanced datasets and one fixed split limit generalization claims; test accuracy is unknown.',
                'Audio was decoded from Opus; WAV conversion cannot restore information lost to compression.',
                'Anonymous artist IDs are unavailable for inner CV grouping; official splits remain unchanged.',
                'Pretraining overlap with these recordings cannot be ruled out. No extra labeled audio was added.',
                'Pooling removes event order. TinyMU uses only the first 10 seconds, while classifier features cover 30 seconds.',
                'ALM candidate scores depend on prompt and tokenization and are not calibrated class probabilities.',
            ]},
        'alm': {'model_id': cfg.get('model', 'TinyMU'), 'split': cfg['split'],
                'prompts': [{'id': prompt_id,
                             'text': '\n\n'.join(f'{key}:\n{prompt}' for key,prompt in cfg['prompts'].items()),
                             'rationale': cfg['prompt_selection_reason']}],
                'protocol': [cfg['audio_preprocessing'], cfg['score_definition'],
                             'Evaluate all validation recordings in both datasets. Keep the pretrained model frozen.',
                             'Rank all six allowed labels by length-normalized conditional likelihood, including EOS.',
                             'The reported Top-3 is a constrained ranking; it does not measure free-form JSON instruction following.'],
                'invalid_handling': cfg['invalid_output_handling'], 'results': alm_results,
                'notes': alm_notes},
        'references': references}
    validate_payload(payload)
    return payload


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results-dir',required=True,type=Path)
    p.add_argument('--data-audit',required=True,type=Path)
    p.add_argument('--profile',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args()
    value=build(a.results_dir,a.data_audit,a.profile)
    a.output.write_text(json.dumps(value,indent=2))
    print(f'Wrote validated measured results to {a.output}')


if __name__=='__main__':
    main()
