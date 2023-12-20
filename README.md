# CoLoRa
To be published at National Conference on Communications (NCC) 2024.

### Abstract
This paper presents an alternative design for the current state-of-the-art covert channel on top of the LoRa physical layer (PHY) to embed covert information over ongoing LoRa transmission. LoRa utilizes the chirp spread spectrum (CSS) modulation technique, where the starting frequency of each chirp is used to encode messages as LoRa symbols. Our proposed covert channel, CoLoRa, allows covert transmission over the LoRa PHY without affecting LoRa chirp frequencies. The key challenges in designing such protocols are also discussed. CoLoRa uses adaptive $2^{m}$ multilevel amplitude modulation to embed covert messages over the ongoing LoRa transmission, where m indicates the modulation index, i.e., the number of bits per covert symbol. The impact of such a covert channel on the communication range of legitimate LoRa transmissions has been discussed and is set at an acceptable (and configurable) level. Furthermore, the performance of the CoLoRa covert communication is investigated in terms of the bit error rate (BER) under different channel conditions. The simulation results for CoLoRa indicate that covert communication achieves a BER of approximately 20% (with m = 2) when the covert transmitter and receiver are 600 m apart, with a target signal-to-noise ratio (SNR) of 6 dB. By increasing the modulation index, the adaptive multilevel amplitude modulation feature effectively improves the BER.

# Simulator requirements
Name          | Version
------------- | -------------
Python        | 3.9
Numpy         | 1.23 and above

# Running the Simulator -- *Simulator.py*
![Images/1.png](Images/1.png?raw=true)

Use the following command on terminal:
> python Simulator.py MODUALTION_INDEX NUMBER_OF_MESSAGES SNR DISTANCE


# Reproducing Results


> [!NOTE]
> To Reproduce exact results use NM $\geq 10 ^{4}$.