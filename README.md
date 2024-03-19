## PolyMerge: A Novel Technique aimed at Dynamic HD Map Updates Leveraging Polylines
https://arxiv.org/abs/2310.18416


Currently, High-Definition (HD) maps are a prerequisite for the stable operation of autonomous vehicles. Such maps contain information about all static road objects for the vehicle to consider during navigation, such as road edges, road lanes, crosswalks, and etc. To generate such an HD map, current approaches need to process pre-recorded environment data obtained from onboard sensors. However, recording such a dataset often requires a lot of time and effort. In addition, every time actual road environments are changed, a new dataset should be recorded to generate a relevant HD map.
This paper addresses a novel approach that allows to continuously generate or update the HD map using onboard sensor data. When there is no need to pre-record the dataset, updating the HD map can be run in parallel with the main autonomous vehicle navigation pipeline.
The proposed approach utilizes the VectorMapNet framework to generate vector road object instances from a sensor data scan. The PolyMerge technique is aimed to merge new instances into previous ones, mitigating detection errors and, therefore, generating or updating the HD map.
The performance of the algorithm was confirmed by comparison with ground truth on the NuScenes dataset. Experimental results showed that the mean error for different levels of environment complexity was comparable to the VectorMapNet single instance error.

Authors: Mohamed Sayed, Stepan Perminov, Dzmitry Tsetserukou
