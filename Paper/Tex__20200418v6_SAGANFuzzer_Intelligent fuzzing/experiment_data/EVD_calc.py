class model_EVD:
    ARTC = []
    VDE = []
    DGC = []
    EVD = []
    def __init__(self):
        self.ARTC = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.VDE = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.DGC = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.EVD = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# model 初始化
GPF = model_EVD()
CNN_1D = model_EVD()
LSTM = model_EVD()
WGAN = model_EVD()
SAGAN = model_EVD()

# model 参数赋值
GPF.ARTC = [0.60	,0.60	,0.60	,0.60	,0.60,	0.60,	0.60,	0.60	,0.60,	0.60]

for i in range(10):
    ARTC_min = min(GPF.ARTC[i], CNN_1D.ARTC[i], LSTM.ARTC[i], WGAN.ARTC[i], SAGAN.ARTC[i])
    ARTC_max = max(GPF.ARTC[i], CNN_1D.ARTC[i], LSTM.ARTC[i], WGAN.ARTC[i], SAGAN.ARTC[i])
    VDE_min = min(GPF.VDE[i], CNN_1D.VDE[i], LSTM.VDE[i], WGAN.VDE[i], SAGAN.VDE[i])
    VDE_max = max(GPF.VDE[i], CNN_1D.VDE[i], LSTM.VDE[i], WGAN.VDE[i], SAGAN.VDE[i])
    DGC_min = min(GPF.DGC[i], CNN_1D.DGC[i], LSTM.DGC[i], WGAN.DGC[i], SAGAN.DGC[i])
    DGC_max = max(GPF.DGC[i], CNN_1D.DGC[i], LSTM.DGC[i], WGAN.DGC[i], SAGAN.DGC[i])

    GPF.EVD[i] = ((GPF.ARTC[i] - ARTC_min) / (ARTC_max - ARTC_min)) + ((GPF.VDE[i] - VDE_min) / (VDE_max - VDE_min)) + (
            (GPF.DGC[i] - DGC_min) / (DGC_max - DGC_min))
    CNN_1D.EVD[i] = ((CNN_1D.ARTC[i] - ARTC_min) / (ARTC_max - ARTC_min)) + ((CNN_1D.VDE[i] - VDE_min) / (VDE_max - VDE_min)) + (
                (CNN_1D.DGC[i] - DGC_min) / (DGC_max - DGC_min))
    LSTM.EVD[i] = ((LSTM.ARTC[i] - ARTC_min) / (ARTC_max - ARTC_min)) + ((LSTM.VDE[i] - VDE_min) / (VDE_max - VDE_min)) + (
                (LSTM.DGC[i] - DGC_min) / (DGC_max - DGC_min))
    WGAN.EVD[i] = ((WGAN.ARTC[i] - ARTC_min) / (ARTC_max - ARTC_min)) + ((WGAN.VDE[i] - VDE_min) / (VDE_max - VDE_min)) + (
                (WGAN.DGC[i] - DGC_min) / (DGC_max - DGC_min))
    SAGAN.EVD[i] = ((SAGAN.ARTC[i] - ARTC_min) / (ARTC_max - ARTC_min)) + ((SAGAN.VDE[i] - VDE_min) / (VDE_max - VDE_min)) + (
                (SAGAN.DGC[i] - DGC_min) / (DGC_max - DGC_min))

print("GPF EVD : " + str(GPF.EVD))
print("CNN EVD : " + str(CNN_1D.EVD))
print("LSTM EVD : " + str(LSTM.EVD))
print("WGAN EVD : " + str(WGAN.EVD))
print("SAGAN EVD : " + str(SAGAN.EVD))
