###] GrappaNet, one of the great work on MRI reconstruction was published in 2020. We tried to implement it since code was not available. The paper can be found here https://arxiv.org/pdf/1910.12325v4.pdf. 

### The implementation might not be exactly same as the author described in the paper due to the lack of proper understanding of various parameters and internal details. I would be glad if someone takes little time and verify my implementation. The MRI community will be benefitted from your input and help.  


Our Dataset is small and acquired with different imaging conditions than fastMRI dataset. This example has been shown for MRIs acquired with 9 coils. Unlike GrappNet paper we have precalculated grappa weights and reused during training. However, Model was trained in eager mode and consumes significant amount of time and GPU Memory for each epoch. We trained our model on 80 GB Apollo GPU
