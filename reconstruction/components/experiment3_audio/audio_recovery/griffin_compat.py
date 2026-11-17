
from typing import Optional
import torch
from torch import Tensor
def _get_complex_dtype(real_dtype: torch.dtype):

    if real_dtype == torch.double:

        return torch.cdouble

    if real_dtype == torch.float:

        return torch.cfloat

    if real_dtype == torch.half:

        return torch.complex32

    raise ValueError(f"Unexpected dtype {real_dtype}")
def griffinlim(

    specgram: Tensor,

    window: Tensor,

    n_fft: int,

    hop_length: int,

    win_length: int,

    power: float,

    n_iter: int,

    momentum: float,

    length: Optional[int],

    rand_init: bool,

) -> Tensor:

    



























































    if not 0 <= momentum < 1:

        raise ValueError("momentum must be in range [0, 1). Found: {}".format(momentum))



    momentum = momentum / (1 + momentum)



                

    shape = specgram.size()

    specgram = specgram.reshape([-1] + list(shape[-2:]))



    specgram = specgram.pow(1 / power)



                          

    if rand_init:

        angles = torch.rand(specgram.size(), dtype=_get_complex_dtype(specgram.dtype), device=specgram.device)

    else:

        angles = torch.full(specgram.size(), 1, dtype=_get_complex_dtype(specgram.dtype), device=specgram.device)



                                              

    tprev = torch.tensor(0.0, dtype=specgram.dtype, device=specgram.device)

    for _ in range(n_iter):

                                                        

        inverse = torch.istft(

            specgram * angles, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, length=length

        )



                                 

        rebuilt = torch.stft(

            input=inverse,

            n_fft=n_fft,

            hop_length=hop_length,

            win_length=win_length,

            window=window,

            center=True,

            pad_mode="reflect",

            normalized=False,

            onesided=True,

            return_complex=True,

        )



                                    

        angles = rebuilt

        if momentum:

            angles = angles - tprev.mul_(momentum)

        angles = angles.div(angles.abs().add(1e-16))



                                    

        tprev = rebuilt



                                      

    waveform = torch.istft(

        specgram * angles, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, length=length

    )



                  

    waveform = waveform.reshape(shape[:-2] + waveform.shape[-1:])



    return waveform

