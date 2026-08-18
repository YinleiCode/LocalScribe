#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from typing import Any
from scribe_py.core.sensevoice_recovery import parse_paraformer_native_timestamps, build_paraformer_native_ownership
from scribe_py.core.transcriber_funasr import FunASRTranscriber, _speech_coverage_diagnostics
from scribe_py.core.types import Segment, TranscribeOptions

def safe(value: Any) -> Any:
    if isinstance(value,dict): return {str(k):safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [safe(v) for v in value]
    if isinstance(value,(str,int,float,bool)) or value is None:return value
    if hasattr(value,'item'):
        try:return safe(value.item())
        except Exception:pass
    if hasattr(value,'tolist'):
        try:return safe(value.tolist())
        except Exception:pass
    return str(value)

def main()->int:
    p=argparse.ArgumentParser(); p.add_argument('--audio',type=Path,required=True); p.add_argument('--transcript',type=Path,required=True); p.add_argument('--out',type=Path,required=True); p.add_argument('--model',required=True); a=p.parse_args()
    audio_hash=hashlib.sha256(a.audio.read_bytes()).hexdigest()
    expected='1dd1abce2f2798c4da730f791f341875341af50dd660d4642e6fd12e734bb94c'
    if audio_hash!=expected: raise SystemExit(f'audio sha256 mismatch: {audio_hash}')
    artifact=json.loads(a.transcript.read_text()); coverage=artifact['filter_stats']['speech_coverage']; segments=artifact['segments']; duration=float(artifact['duration'])
    if len(segments)!=139: raise SystemExit(f'segment count mismatch: {len(segments)}')
    transcriber=FunASRTranscriber(backend_name='funasr'); model=transcriber._load(a.model); options=TranscribeOptions(model_id=a.model,language='zh',audio_preprocess='off')
    raw=transcriber._generate(model,a.audio,options,sensevoice=False); items=raw if isinstance(raw,list) else [raw]; raw_safe=safe(items)
    parsed=parse_paraformer_native_timestamps(raw_safe,duration_s=duration)
    failed=[w for w in coverage['strict_probe_windows'] if w.get('status')=='failed']
    final_text=''.join(str(s.get('text') or '') for s in segments)
    ownership=build_paraformer_native_ownership(final_text=final_text,native_units=parsed['units'],strict_windows=failed)
    accepted={(round(c['core_start'],3),round(c['core_end'],3)) for c in ownership['claims']}
    recognized=[(float(w['core_start']),float(w['core_end'])) for w in coverage['strict_probe_windows'] if w.get('status')=='recognized' or (round(float(w['core_start']),3),round(float(w['core_end']),3)) in accepted]
    speech=[(float(x['start']),float(x['end'])) for x in coverage['speech_intervals']]
    projected=_speech_coverage_diagnostics(speech,[Segment(start=s,end=e,text='recognized') for s,e in recognized],duration=duration,collar_s=0.0,vad_status='ok',vad_reason='paraformer_native_overlay')
    raw_hash=hashlib.sha256(json.dumps(raw_safe,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    out={'audio_sha256':audio_hash,'segment_count':len(segments),'frozen_text_sha256':coverage.get('local_recovery',{}).get('after',{}).get('text_sha256'),'model':a.model,'raw_items_sha256':raw_hash,'raw_items':raw_safe,'native_evidence':parsed,'ownership':ownership,'projected':projected}
    a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(out,ensure_ascii=False,indent=2)); print(json.dumps({'out':str(a.out),'raw_items':len(raw_safe),'native_chars':parsed['native_character_count'],'rejected_items':len(parsed['rejected_items']),'equal_ratio':ownership['equal_char_ratio'],'claims':len(ownership['claims']),'projected_ratio':projected.get('speech_coverage_ratio'),'projected_max_gap_s':projected.get('max_uncovered_speech_s'),'raw_items_sha256':raw_hash,'claims_sha256':ownership['claims_sha256']},ensure_ascii=False)); return 0
if __name__=='__main__': raise SystemExit(main())
