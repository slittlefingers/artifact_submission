
import json
from importlib import import_module
from pathlib import Path
import numpy as np


class MediaUnavailable(RuntimeError):
    pass


def resolve_rules(profile, tokenizer):
    rules = profile.get('media', {})
    if not rules:
        return {}
    data = json.loads(Path(tokenizer).read_text(encoding='utf-8'))
    tokens = {t['content']: t['id'] for t in data['added_tokens'] if t.get('special')}
    resolved = {}
    used = set()
    for modality, rule in rules.items():
        if modality not in ('image', 'audio') or set(rule) != {'start', 'end', 'soft', 'handler'}:
            raise ValueError('Media rules require image/audio with start, end, soft, handler')
        entry = {'handler': rule['handler']}
        if not isinstance(entry['handler'], str) or ':' not in entry['handler']:
            raise ValueError('Media handler must be a trusted module:function')
        for key in ('start', 'end', 'soft'):
            if rule[key] not in tokens:
                raise ValueError(f'Missing special media token: {rule[key]}')
            token = tokens[rule[key]]
            if token in used:
                raise ValueError('Media token roles must be distinct')
            used.add(token)
            entry[key] = token
        resolved[modality] = entry
    return resolved


def validate_spans(spans, positions):
    
    if not isinstance(spans, list):
        raise ValueError('Media spans must be an array')
    last = -1
    available = set(map(int, positions))
    for span in spans:
        if not isinstance(span, dict) or set(span) != {'modality', 'marker_position', 'start', 'end', 'source'}:
            raise ValueError('Media spans require modality, marker_position, start, end, source only')
        if span['modality'] not in ('image', 'audio'):
            raise ValueError('Unknown media modality')
        if any(type(span[k]) is not int for k in ('marker_position', 'start', 'end')):
            raise ValueError('Media positions must be integers')
        marker, start, end = (span[k] for k in ('marker_position', 'start', 'end'))
        if marker < 0 or start != marker + 1 or end <= start or marker <= last:
            raise ValueError('Media spans must be ordered and nonoverlapping; start follows marker')
        if marker not in available or not set(range(start, end + 1)).issubset(available):
            raise ValueError('Observation must include the start marker, whole media span and end marker')
        if not isinstance(span['source'], str) or not span['source'].strip():
            raise ValueError('Document how media boundaries are known')
        last = end
    return spans


class MediaRouter:
    def __init__(self, rules, spans, positions, locator=None, handlers=None):
        self.rules = rules
        self.locator = locator
        self.handlers = handlers or {}
        self.spans = {s['marker_position']: s for s in validate_spans(spans, positions)}
        if any(s['modality'] not in rules for s in spans):
            raise ValueError('Media span has no selected model rule')

    def kind(self, token, role='start'):
        return next((kind for kind, rule in self.rules.items() if rule.get(role) == token), None)

    def finish_end(self,backend,segment,target,config,output):
        modality=segment['modality'];position=segment['end']
        finish=getattr(self.handlers.get(modality),'finish_end',None)
        if not callable(finish):raise MediaUnavailable('continuous_media_end_handler_unavailable')
        result=finish(backend,dict(modality=modality,position=position,target=np.asarray(target).copy(),
            config=config,output=Path(output)))
        if result.get('status')!='complete' or backend.position!=position+1:
            raise MediaUnavailable('continuous_media_end_not_advanced')
        return result

    def recover(self, backend, token, marker_position, targets, positions, config, output):
        modality = self.kind(token)
        if modality is None:
            raise ValueError('Only accepted media start tokens may dispatch')
        span = self.spans.get(marker_position)
        boundary = None
        if span is None and self.locator is not None:
            span,boundary = self.locator.locate(modality,marker_position,targets,positions,config)
            validate_spans([span],positions)
        if span is None:
            raise MediaUnavailable('media_boundary_unknown: start token does not reveal media length')
        if span['modality'] != modality:
            raise MediaUnavailable('media_boundary_mismatch: recovered marker disagrees with structure')
        indices = np.flatnonzero((positions >= span['start']) & (positions < span['end']))
        handler = self.handlers.get(modality)
        if handler is None:
            module, name = self.rules[modality]['handler'].split(':', 1)
            handler = getattr(import_module(module), name)
        request = dict(modality=modality, span=dict(span), positions=positions[indices].copy(),
                       targets=np.asarray(targets)[indices].copy(), config=config, output=Path(output),
                       boundary=boundary, end_target=np.asarray(targets)[np.searchsorted(positions,span['end'])].copy())
        result = handler(backend, request)
        resume=span['end']+int(bool(result.get('end_consumed',False))) if isinstance(result,dict) else span['end']
        if not isinstance(result, dict) or result.get('status') != 'complete' or backend.position != resume:
            raise MediaUnavailable('media_handler_incomplete: backend has not advanced to the end marker')
        return dict(result, modality=modality, start=span['start'], end=span['end'],
                    resume_position=resume, boundary_proposal=boundary,
                    handler=self.rules[modality]['handler'], boundary_source=span['source'])
