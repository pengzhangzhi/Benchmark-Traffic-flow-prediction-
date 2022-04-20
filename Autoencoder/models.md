model1
struttura di partenza, vedere my_model.png

model2
come model1, ma utilzza lstm tra encoder e decoder

model3
mantiene struttura encoder-decoder ma utilizza il TimeDistributed layer nell'
encoder (come dovrebbe essere fatto nel paper di riferimento)
tra encoder e decoder utilizza cmu (da "PredCNN: Predictive Learning with Cascade Convolutions")

model4
come model1, ma utilizza lstm dopo ogni convoluzione nell'encoder

model5
come model2, ma utilizza residui in ogni convoluzione