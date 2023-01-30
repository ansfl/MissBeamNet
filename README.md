# MissBeamNet: Learning Missing Doppler Velocity Log Beam Measurements


## Introduction

One of the primary means of sea exploration is autonomous underwater vehicles (AUVs). To perform these tasks, AUVs must navigate the rough challenging sea environment. AUVs usually employ an inertial navigation system (INS), aided by a Doppler velocity log (DVL), to provide the required navigation accuracy. The DVL transmits four acoustic beams to the seafloor, and by measuring changes in the frequency of the returning beams, the DVL can estimate the AUV velocity vector. However, in practical scenarios, not all the beams are successfully reflected. When only three beams are available, the accuracy of the velocity vector is degraded. When fewer than three beams are reflected, the DVL cannot estimate the AUV velocity vector. This paper presents a data-driven approach, MissBeamNet, to regress the missing beams in partial DVL beam measurement cases. To that end, a deep neural network (DNN) model is designed to process the available beams along with past DVL measurements to regress the missing beams. The AUV velocity vector is estimated using the available measured and regressed beams. To validate the proposed approach, sea experiments were made with the "Snapir" AUV, resulting in an 11 hours dataset of DVL measurements. Our results show that the proposed system can accurately estimate velocity vectors in situations of missing beam measurements.

## Dataset and Code

Coming soon


![Alt text](/auvpic.jpg "The ”Snapir” being pulled out of the water after a successful mission.")



## Citation
If you found the paper's methods, data, or code helpful in your research, please cite our paper:

    @article{Yona2023Miss,
    title = {MissBeamNet: Learning Missing Doppler Velocity Log Beam Measurements},
    year = {2023},
    journal={arXiv preprint arXiv:2301.11597},
    url = {https://doi.org/10.48550/arXiv.2301.11597},
    author = {Mor Yona and Itzik Klein},
    }

