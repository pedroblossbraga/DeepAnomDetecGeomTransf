# DeepAnomDetecGeomTransf

Experiments and extensions to the method provided in "Deep Anomaly Detection using Geometric Transformations" [1]

## Experiments
- New image transformations
    - quantile histogram equalization
    - color jitter
    - zoom / crop
- Alternative normality scoring, via Shannon Entropy
- Spatial Localization of Features
    - Averaging activation maps of relevant convolutional or activation layers
    - Grad-CAM: weighting 2D-activations with the average gradient
- Uncertainty Analysis
    - Monte Carlo Dropout

## References

        [1] I. Golan and R. El-Yaniv, in Advances in Neural Information Processing Sys-
        tems, edited by S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-
        Bianchi, and R. Garnett (Curran Associates, Inc., 2018), vol. 31.