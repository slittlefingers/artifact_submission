
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer

ROOT=Path(__file__).resolve().parent;CODE=ROOT.parent

def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,data):Path(path).write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True,help='Recovery output directory');args=parser.parse_args()
    folder=Path(args.output);protocol=read(folder/'protocol.json');result=read(folder/'recovery/result.json')
    if result['status']!='complete':raise ValueError('This evaluation expects a completed recovery')
    source=Path(protocol['source_cache']);assert sha(source)==protocol['source_sha256']
    data=source.parent;sample=read(data/'sample.json')
    labels_root=data.parent.parent/'evaluation'/data.name
    labels_path=labels_root/f"turn{protocol['turns']}_tokens.json"
    labels=read(labels_path)['all_runtime_ids']
    with np.load(folder/'observation.npz',allow_pickle=False) as z:positions=set(map(int,z['positions']))
    trace_path=folder/'recovery/trace.jsonl'
    trace=[json.loads(line) for line in trace_path.read_text(encoding='utf8').splitlines()]
    assert len({r['position'] for r in trace})==len(trace)
    predicted={r['position']:r['accepted_token_id'] for r in trace if r.get('accepted_token_id') is not None}
    stored=dict(zip(result['recovered_token_positions'],result['recovered_token_ids']))
    assert predicted==stored
    expected={p for p in positions if labels[p]>=0}
    tok=Tokenizer.from_file(str(CODE/'npu_model/gemma-4-E2B-it/tokenizer.json'))
    special={
        token_id for token_id,t in tok.get_added_tokens_decoder().items() if t.special}
    user=set();assistant=set();turn_positions={};turn_roles={}
    for turn in sample['turns']:
        number=turn['turn'];start=turn['before_step'];end=turn['state']['step']
        tokens=read(labels_root/f'turn{number}_tokens.json')
        assert tokens['all_runtime_ids']==labels[:len(tokens['all_runtime_ids'])]
        body=tok.encode(turn['user']['text'],add_special_tokens=False).ids
                                                                                
                                                                            
        hits=[i for i in range(start,end-len(body)+1) if labels[i:i+len(body)]==body]
        if len(hits)!=1:raise ValueError(f'User text cannot be uniquely located in turn {number}')
        up=set(range(hits[0],hits[0]+len(body))) & expected
        ap=set(tokens['assistant_positions']) & expected
        ap={p for p in ap if labels[p] not in special}
        user|=up;assistant|=ap
        turn_positions[number]=set(range(start,end)) & expected
        turn_roles[number]=(up,ap)
    body=user|assistant;formatting=expected-body
    def score(selection):
        selection=set(selection);correct=sum(predicted.get(p)==labels[p] for p in selection)
        return dict(correct=correct,total=len(selection),missing=sum(p not in predicted for p in selection),
                    accuracy_percent=100*correct/len(selection) if selection else None)
    boundaries=[]
    for segment in result['media_segments']:
        interior=-1 if segment['modality']=='image' else -2
        ending=-3 if segment['modality']=='image' else -4
        true_ends=[p for p in positions if labels[p]==ending]
        end=segment['end'];start=segment['start']
        boundaries.append(dict(modality=segment['modality'],start=start,end=end,
            end_matches=end in true_ends,
            interior_matches=all(labels[p]==interior for p in range(start,end)),tokens=end-start))
    expected_media=[]
    for p in sorted(positions):
        if labels[p] not in (-1,-2) or (p-1 in positions and labels[p-1]==labels[p]):continue
        end=p
        while end in positions and labels[end]==labels[p]:end+=1
        kind='image' if labels[p]==-1 else 'audio'
        matching=[s for s in result['media_segments'] if s['modality']==kind and s['start']==p and s['end']==end]
        expected_media.append(dict(modality=kind,start=p,end=end,marker_position=p-1,
            tokens=end-p,recovered_with_exact_boundaries=len(matching)==1))
    extra_text=sorted(p for p in predicted if p in positions and labels[p]<0)
    media_coverage=dict(expected_segments=len(expected_media),predicted_segments=len(result['media_segments']),
        exact_segments=sum(s['recovered_with_exact_boundaries'] for s in expected_media),
        missing_or_misaligned=[s for s in expected_media if not s['recovered_with_exact_boundaries']],
        text_predictions_on_continuous_media_positions=len(extra_text),
        extra_text_positions=extra_text,expected=expected_media)
    media_coverage['structure_correct']=(media_coverage['exact_segments']==len(expected_media)==len(result['media_segments']) and not extra_text)
    errors=[]
    for p in sorted(expected):
        guessed=predicted.get(p)
        if guessed!=labels[p]:errors.append(dict(position=p,expected_id=labels[p],predicted_id=guessed,
            expected=tok.decode([labels[p]],skip_special_tokens=False),
            predicted=tok.decode([guessed],skip_special_tokens=False) if guessed is not None else None))
    report=dict(case=protocol['case'],run=folder.name,metric='Exact native token-ID match at original cache positions',
        variant=protocol.get('variant','multimodal'),
        offline_only=True,labels_used_during_recovery=False,
        all_vocabulary_tokens=score(expected),text_body=score(body),user_body=score(user),
        assistant_body=score(assistant),template_separators_and_markers=score(formatting),
        per_turn=[dict(turn=n,all_vocabulary_tokens=score(ps),user_body=score(turn_roles[n][0]),
                      assistant_body=score(turn_roles[n][1])) for n,ps in turn_positions.items()],
        media_boundaries=boundaries,media_coverage=media_coverage,errors=errors,
        excluded=dict(public_bos=protocol['public_prefix_ids'],
            image_interior=sum(labels[p]==-1 for p in positions),
            audio_interior=sum(labels[p]==-2 for p in positions),
            continuous_media_ends=sum(labels[p] in (-3,-4) for p in positions),
            final_pending_token_positions=list(range(max(positions)+1,len(labels)))),
        acceptance=dict(threshold_passed=sum(bool(r.get('threshold_passed')) for r in trace if r['position'] in expected),
                        approximate=sum(bool(r.get('approximate')) for r in trace if r['position'] in expected)),
        provenance=dict(labels=str(labels_path),labels_sha256=sha(labels_path),trace_sha256=sha(trace_path),
                        result_sha256=sha(folder/'recovery/result.json'),source_sha256=sha(source)),
        limitation='Vocabulary accuracy excludes continuous media positions; inspect media_coverage for missed media and spurious text. Complete means traversal finished, not correct multimodal recovery.')
    if protocol.get('variant')=='token_surrogate':
        report['limitation']='This ablation deliberately substitutes vocabulary tokens at continuous media positions. They have no true vocabulary ID and are excluded from text accuracy. No image or audio artifact is reconstructed; media_coverage reports this deliberate absence. Native text accuracy remains position-aligned.'
    save(folder/'evaluation.json',report)
    print(json.dumps({k:report[k] for k in ('all_vocabulary_tokens','text_body','user_body','assistant_body',
        'template_separators_and_markers','acceptance','media_boundaries','excluded')},indent=2))

if __name__=='__main__':main()
