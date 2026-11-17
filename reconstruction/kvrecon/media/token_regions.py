






from dataclasses import replace
from pathlib import Path
import numpy as np
from . import MediaUnavailable
from ..common import dump


class TokenSurrogateRegionPolicy:
    def __init__(self, text_config, targets, positions, locator, marker_ids=None,
                 output=None, media_epsilon=2.0):
        text_config.validate()
        if (not text_config.require_stagnation_for_acceptance or
                text_config.ridge_stagnation_rounds is None or
                text_config.stagnation_neighbors < 1):
            raise ValueError('Text regions require gated ridge and neighborhood acceptance')
        if not np.isfinite(media_epsilon) or media_epsilon <= 0:
            raise ValueError('Positive media fast-accept epsilon required')
        self.positions = np.asarray(positions)
        self.targets = np.asarray(targets)
        if (self.positions.ndim != 1 or not len(self.positions) or
                self.positions.dtype.kind not in 'iu' or
                not np.array_equal(self.positions,
                    np.arange(self.positions[0], self.positions[0]+len(self.positions)))):
            raise ValueError('Token-region policy requires nonempty contiguous integer positions')
        if (self.targets.ndim != 2 or len(self.targets) != len(self.positions) or
                not self.targets.shape[1] or not np.isfinite(self.targets).all()):
            raise ValueError('Malformed observed K/V targets')
        if not callable(getattr(locator, 'locate', None)):
            raise ValueError('A public-reference end locator is required')
        marker_ids = {'image': 255999, 'audio': 256000} if marker_ids is None else dict(marker_ids)
        if (not marker_ids or not set(marker_ids) <= {'image', 'audio'} or
                any(type(t) is not int or t < 0 for t in marker_ids.values()) or
                len(set(marker_ids.values())) != len(marker_ids)):
            raise ValueError('Distinct public image/audio start token IDs required')
        self.markers = {token: kind for kind, token in marker_ids.items()}
        self.locator = locator
        self.text_config = text_config
        self.media_config = replace(text_config, epsilon=float(media_epsilon),
                                    require_stagnation_for_acceptance=False).validate()
        self.output = Path(output) if output is not None else None
        self.regions = []
        self.active = None
        self.mode = 'text'
        self.position = None
        self._last_call = None
        self._last_config = None
        self.error = None
        self._save()

    @property
    def status(self):
        return dict(mode=self.mode, position=self.position, regions=self.regions,
            active_region=self.active, text_threshold=self.text_config.epsilon,
            media_fast_accept_threshold=self.media_config.epsilon,
            media_distance_gate=False, labels_used=False, neural_media_models_used=False,
            boundary_verified=False, error=self.error,
            note='Media regions are inferred from accepted start markers and observed V. '
                 'Every position including continuous ends is replayed as a vocabulary token.')

    def _save(self):
        if self.output is not None:
            self.output.parent.mkdir(parents=True, exist_ok=True)
            dump(self.output, self.status)

    def __call__(self, position, previous_token_id, previous_position):
        if not isinstance(position, (int, np.integer)):
            raise ValueError('Integer observed position required')
        position = int(position)
        if not self.positions[0] <= position <= self.positions[-1]:
            raise ValueError('Requested position lies outside the observation')
        if previous_token_id is not None and (not isinstance(previous_token_id, (int, np.integer)) or previous_token_id < 0):
            raise ValueError('Previous accepted token ID must be nonnegative')
        if previous_position is not None and (not isinstance(previous_position, (int, np.integer)) or int(previous_position) != position-1):
            raise ValueError('Previous accepted position must immediately precede this one')
        if (previous_position is None) != (previous_token_id is None):
            raise ValueError('Previous token and position must both be supplied or omitted')
        call = (position, previous_token_id, previous_position)
        if self._last_call == call:
            return self._last_config
        if self.position is not None and position != self.position+1:
            raise ValueError('Token-region policy must be called in sequential position order')
        if self.position is None and position != self.positions[0]:
            raise ValueError('Token-region policy must begin at the first observed position')

                                                                                  
        previous_was_media = (previous_position is not None and any(
            region['start'] <= previous_position <= region['end'] for region in self.regions))
        if self.active is not None and position > self.active['end']:
            self.active = None
        if self.active is None and not previous_was_media and previous_token_id in self.markers:
            kind = self.markers[previous_token_id]
            try:
                span, audit = self.locator.locate(kind, int(previous_position),
                    self.targets, self.positions, self.text_config)
                start, end = span['start'], span['end']
                if (span.get('modality') != kind or span.get('marker_position') != previous_position or
                        start != position or not isinstance(end, (int, np.integer)) or
                        not start < end <= self.positions[-1]):
                    raise MediaUnavailable('invalid_surrogate_media_boundary_proposal')
                region = dict(modality=kind, marker_position=int(previous_position),
                    marker_token_id=int(previous_token_id), start=int(start), end=int(end),
                    interior_tokens=int(end-start), end_inclusive_for_threshold=True,
                    source=span.get('source'), boundary_proposal=audit,
                    scan_margin=(audit['runner_up_distance']-audit['distance'])
                        if audit.get('runner_up_distance') is not None else None,
                    verified=False)
                self.regions.append(region)
                self.active = region
            except (MediaUnavailable, ValueError) as error:
                self.error = str(error)
                self.mode = 'unresolved_boundary'
                self.position = position
                self._save()
                raise
        self.position = position
        self.mode = self.active['modality'] if self.active is not None else 'text'
        result = self.media_config if self.active is not None else self.text_config
        self._last_call, self._last_config = call, result
        self._save()
        return result
