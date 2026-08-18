#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
from scribe_py.core.sensevoice_recovery import build_anchor_character_ownership
from scribe_py.core.transcriber_funasr import FunASRTranscriber, _speech_coverage_diagnostics
from scribe_py.core.types import Segment, TranscribeOptions

def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument('--audio',type=Path,required=True); p.add_argument('--transcript',type=Path,required=True); p.add_argument('--out',type=Path,required=True); p.add_argument('--model',default='iic/SenseVoiceSmall'); a=p.parse_args()
    artifact=json.loads(a.transcript.read_text()); coverage=artifact['filter_stats']['speech_coverage']; segments=artifact['segments']
    transcriber=FunASRTranscriber(backend_name='sensevoice'); model=transcriber._load(a.model)
    anchors,_language,stats=transcriber._run_sensevoice_wallclock_vad(model,a.audio,TranscribeOptions(model_id=a.model,language='zh',audio_preprocess='off'),None)
    anchor_chunks=[{'start':s.start,'end':s.end,'text':s.text,'status':'recognized'} for s in anchors]
    failed=[w for w in coverage['strict_probe_windows'] if w.get('status')=='failed']
    ownership=build_anchor_character_ownership(final_text=''.join(str(s.get('text') or '') for s in segments),anchor_chunks=anchor_chunks,strict_windows=failed)
    accepted={(round(c['core_start'],3),round(c['core_end'],3)) for c in ownership['claims']}
    recognized=[(float(w['core_start']),float(w['core_end'])) for w in coverage['strict_probe_windows'] if w.get('status')=='recognized' or (round(float(w['core_start']),3),round(float(w['core_end']),3)) in accepted]
    speech=[(float(x['start']),float(x['end'])) for x in coverage['speech_intervals']]
    projected=_speech_coverage_diagnostics(speech,[Segment(start=s,end=e,text='recognized') for s,e in recognized],duration=float(artifact['duration']),collar_s=0.0,vad_status='ok',vad_reason='anchor_ownership_replay')
    out={'anchor_stats':stats,'anchor_segments':len(anchors),'ownership':ownership,'projected':projected,'frozen_text_sha256':coverage.get('local_recovery',{}).get('after',{}).get('text_sha256'),'segment_count':len(segments)}
    a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(out,ensure_ascii=False,indent=2)); print(json.dumps({'out':str(a.out),'anchor_segments':len(anchors),'equal_ratio':ownership['equal_char_ratio'],'claims':len(ownership['claims']),'projected_ratio':projected.get('speech_coverage_ratio'),'projected_max_gap_s':projected.get('max_uncovered_speech_s'),'claims_sha256':ownership['claims_sha256']},ensure_ascii=False)); return 0
if __name__=='__main__': raise SystemExit(main())
