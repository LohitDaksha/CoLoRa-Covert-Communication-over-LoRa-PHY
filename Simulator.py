import numpy as np
import random
import copy
import matplotlib.pyplot as plt

import sys

MESSAGES = []

def GeneratePayload(N):
    for _ in range(N):
        MESSAGES.append(list(np.random.randint(low = 0,high=2,size=111)))


modulation_index = eval(sys.argv[1])
Number_of_Messages = eval(sys.argv[2])
SNR = eval(sys.argv[3])
Distance = eval(sys.argv[4])

Adaptive_Modulation = False

header_len = 60             
header = list(np.random.randint(low = 0,high=2,size=60))    #[0]*54 + [1]*6

payload_len = eval("117 - 9*modulation_index")
CRC_code_len = 4
CRC_code = [1,1,0,1]
np.pi = np.pi

Total_lora_symbol = 180
samples_per_symbol = 128

SF = 7

sample_per_lora_symbol = 2**SF

Lora_symbol_generation = np.array([])
I_samples_original = np.array([])
Q_samples_original = np.array([])


def Embed_CRC(_header, _payload_TX):
    _D = list(_header) + list(_payload_TX) + [0]*(CRC_code_len - 1)
    
    _i = 0
    while _i< header_len + payload_len:
        _j = 0
        while _j<CRC_code_len:
            _D[_i+_j] = _D[_i+_j] ^ CRC_code[_j];
            _j += 1
        
        while (_i<header_len+payload_len+CRC_code_len-1) and (_D[_i]!=1):
            _i += 1
    
    for _i in range(CRC_code_len - 1):
        _payload_TX += [_D[header_len+payload_len+_i]]
    
    del _D


def Embed(msg_id):
    A_max = 1.58
    A_min = 1.278
    if Adaptive_Modulation:
        A_min=1.278
    else:
        A_min = (A_max+A_min)/2
        
    diff = (A_max - A_min)/(2**modulation_index - 1)
    
    preamble_len = 9
        
    _SubPreamble = [A_max]*128 + [A_max - diff]*128 + [A_min]*128
    _Preamble = _SubPreamble * 3
    
    _Header = []
    for i in range(0, header_len, modulation_index):
        _msg = ''
        for j in range(modulation_index):
            _msg += str(header[i+j])
        _S = eval('0b'+_msg)
        
        _Header += [A_min + _S*diff]*128
    
    _Payload = [] 
    _PrePayload = copy.deepcopy(MESSAGES[msg_id][:117 - 9*modulation_index])
    
    Embed_CRC(_header = header, _payload_TX = _PrePayload)
    
    for i in range(0, len(_PrePayload), modulation_index):
        _msg = ''
        for j in range(modulation_index):
            _msg += str(_PrePayload[i+j])
        _S = eval('0b'+_msg)
        
        _Payload += [A_min + _S*diff]*128
        
    Embedded_Msg = []
    
    for ct in range(modulation_index):
        Embedded_Msg += _Preamble + _Header + _Payload
        
        
    # plt.plot(Embedded_Msg)
    # plt.show()
    
    _I_TX = (1/A_max)*I_samples_original*Embedded_Msg
    _Q_TX = (1/A_max)*Q_samples_original*Embedded_Msg
    
    return Embedded_Msg, _I_TX, _Q_TX  


def CovertRX(I_RX, Q_RX):
    Sucess = True
    Received_RX = list((np.array(I_RX)**2 + np.array(Q_RX)**2)**0.5)
    
    # plt.plot(Received_RX)
    # plt.show()
    
    _SubPreambleRX_1 = Received_RX[:9*samples_per_symbol]
    _Amax = np.mean(_SubPreambleRX_1[0*samples_per_symbol:1*samples_per_symbol] + _SubPreambleRX_1[3*samples_per_symbol:4*samples_per_symbol] + _SubPreambleRX_1[6*samples_per_symbol:7*samples_per_symbol])
    _Amin = np.mean(_SubPreambleRX_1[2*samples_per_symbol:3*samples_per_symbol] + _SubPreambleRX_1[5*samples_per_symbol:6*samples_per_symbol] + _SubPreambleRX_1[8*samples_per_symbol:9*samples_per_symbol])
    _Amax_2 = np.mean(_SubPreambleRX_1[1*samples_per_symbol:2*samples_per_symbol] + _SubPreambleRX_1[4*samples_per_symbol:5*samples_per_symbol] + _SubPreambleRX_1[7*samples_per_symbol:8*samples_per_symbol]) 
    
    _diff_RX = _Amax - _Amax_2
     
    try:
        _m_RX = np.log2(1 + (_Amax - _Amin)/_diff_RX)
        _m_RX = int(np.rint(_m_RX))
    except:
        return False, -1,-1,-1
    
    if _m_RX!=modulation_index:
        return False, -1,-1,-1
        
    
    
    _HP_RX = np.array([0.0]*int((Total_lora_symbol/_m_RX)-9)*samples_per_symbol)  
    for _i in range(_m_RX):
        _HP_RX += np.array(Received_RX[int((9+(_i)*(Total_lora_symbol/_m_RX))*samples_per_symbol):int((_i+1)*(Total_lora_symbol/_m_RX)*samples_per_symbol)])
    
    _HP_RX = np.array(_HP_RX)
    _HP_RX *= (1/_m_RX)
    
    # plt.plot(_HP_RX)
    # plt.show()
    
    _Header_Payload_CRC_RX = ''
    for _i in range(0, len(_HP_RX), samples_per_symbol):
        _Level = np.mean(_HP_RX[_i:_i+samples_per_symbol])
        try:
            _Symbol = int(np.rint(np.log2(1+ (_Level - _Amin)/_diff_RX)))
        except:
            # print('Too much noise')
            Sucess = False
            return False, -1,-1,-1
        if _Symbol<0:
            _Symbol = 0
        _Header_Payload_CRC_RX += '0'*(_m_RX-len(bin(_Symbol)[2:]))+bin(_Symbol)[2:]
        
    
    _Header = []
    _Payload_CRC = []
    for _i in range(len(_Header_Payload_CRC_RX)):
        if _i>=60:
            _Payload_CRC.append(eval(_Header_Payload_CRC_RX[_i]))
        else:
            _Header.append(eval(_Header_Payload_CRC_RX[_i]))
            
    _Payload_RX = _Payload_CRC[:-3]
    
    try:
        Embed_CRC(_Header, _Payload_RX)    
    except:
        return False, -1,-1,-1
    if(_Payload_RX == _Payload_CRC):
        CRC_Check = True
    else:
        CRC_Check = False
        
    return Sucess, copy.deepcopy(_Header), copy.deepcopy(_Payload_RX[:-3]), CRC_Check

    
def __CovertRX(I_RX, Q_RX):
    Sucess = True
    Received_RX = list((np.array(I_RX)**2 + np.array(Q_RX)**2)**0.5)
    
    # plt.plot(Received_RX)
    # plt.show()
    
    _SubPreambleRX_1 = Received_RX[:9*samples_per_symbol]
    _Amax = np.mean(_SubPreambleRX_1[0*samples_per_symbol:1*samples_per_symbol])
    _Amin = np.mean(_SubPreambleRX_1[2*samples_per_symbol:3*samples_per_symbol])
    _Amax_2 = np.mean(_SubPreambleRX_1[1*samples_per_symbol:2*samples_per_symbol])

    _diff_RX = _Amax - _Amax_2
     
    try:
        _m_RX = np.log2(1 + (_Amax - _Amin)/_diff_RX)
        _m_RX = int(np.rint(_m_RX))
    except:
        return False, -1,-1,-1
    
    if _m_RX!=modulation_index:
        return False, -1,-1,-1
        
    
    
    _HP_RX = np.array([0.0]*int((Total_lora_symbol/_m_RX)-9)*samples_per_symbol)  
    for _i in range(1):
        _HP_RX += np.array(Received_RX[int((9+(_i)*(Total_lora_symbol/_m_RX))*samples_per_symbol):int((_i+1)*(Total_lora_symbol/_m_RX)*samples_per_symbol)])
    
    _HP_RX = np.array(_HP_RX)
    _HP_RX *= (1/1)
    
    # plt.plot(_HP_RX)
    # plt.show()
    
    _Header_Payload_CRC_RX = ''
    for _i in range(0, len(_HP_RX), samples_per_symbol):
        _Level = np.mean(_HP_RX[_i:_i+samples_per_symbol])
        try:
            _Symbol = int(np.rint(np.log2(1+ (_Level - _Amin)/_diff_RX)))
        except:
            Sucess = False
            return False, -1,-1,-1
        if _Symbol<0:
            _Symbol = 0
        _Header_Payload_CRC_RX += '0'*(_m_RX-len(bin(_Symbol)[2:]))+bin(_Symbol)[2:]
        
    
    _Header = []
    _Payload_CRC = []
    for _i in range(len(_Header_Payload_CRC_RX)):
        if _i>=60:
            _Payload_CRC.append(eval(_Header_Payload_CRC_RX[_i]))
        else:
            _Header.append(eval(_Header_Payload_CRC_RX[_i]))
            
    _Payload_RX = _Payload_CRC[:-3]
    
    try:
        Embed_CRC(_Header, _Payload_RX)    
    except:
        return False, -1,-1,-1
    if(_Payload_RX == _Payload_CRC):
        CRC_Check = True
    else:
        CRC_Check = False
        
    return Sucess, copy.deepcopy(_Header), copy.deepcopy(_Payload_RX[:-3]), CRC_Check


def ThermalNoise(I_samples, Q_samples, __SNR=10):
    _I_power = np.mean(I_samples**2)
    _I_noise_power = _I_power/(10**(__SNR/10))
    _I_noise = np.random.normal(0,1,len(I_samples))
    _I_noise = ((_I_noise_power**0.5)/((2*np.mean(_I_noise**2))**0.5)) * _I_noise
    
    _Q_power = np.mean(Q_samples**2)
    _Q_noise_power = _Q_power/(10**(__SNR/10))
    _Q_noise = np.random.normal(0,1,len(Q_samples))
    _Q_noise = ((_Q_noise_power**0.5)/((2*np.mean(_Q_noise**2))**0.5)) * _Q_noise
    
    return np.array(I_samples) + _I_noise, np.array(Q_samples) + _Q_noise 

def Fading_Rayleigh(D, _I_1, _Q_1):
    path_loss_exponent = 2.5
    PL_factor = (3*pow(10, 8)/(4*np.pi*915*pow(10, 6)))/((D**path_loss_exponent)**0.5)
    
    _I_Faded = PL_factor*_I_1
    _Q_Faded = PL_factor*_Q_1
    
    _CombIQ = _I_Faded + 1j*_Q_Faded
    
    z = np.exp(1j * np.random.uniform(0, 2*np.pi, 1))
    
    _I_fading_rayleigh = np.real(_CombIQ*z)
    _Q_fading_rayleigh = np.imag(_CombIQ*z)
    

    return copy.deepcopy(_I_fading_rayleigh), copy.deepcopy(_Q_fading_rayleigh)
       
    
def XOR_LIST(_A, _B):
    count = 0
    if len(_A)!=len(_B):
        pass
    
    for i in range(min(len(_A), len(_B))):
        if _A[i]!=_B[i]:
            count+=1
    
    return count

if __name__ == "__main__":
    
    GeneratePayload(Number_of_Messages)
    
    for i in range(Total_lora_symbol):
        Lora_symbol_generation = np.append(Lora_symbol_generation, [random.randint(0, samples_per_symbol)])
        
        _I = []
        _Q = []
        
        for n in range(sample_per_lora_symbol):
            _I.append(1.58*np.cos(2*np.pi*((n*n)/(2*sample_per_lora_symbol)+((Lora_symbol_generation[i]/sample_per_lora_symbol)-0.5)*n)))
            _Q.append(1.58*np.sin(2*np.pi*((n*n)/(2*sample_per_lora_symbol)+((Lora_symbol_generation[i]/sample_per_lora_symbol)-0.5)*n)))
         
        I_samples_original = np.append(I_samples_original, _I)
        Q_samples_original = np.append(Q_samples_original, _Q)
        
    BER_sum = 0
    Fail = 0
    for MSG in range(Number_of_Messages):
        if (MSG%100==0):
            print("No of messages completed:", MSG)
        
        __, I_afterTX, Q_afterTX = Embed(MSG)
        I_after_Fading_Rayleigh, Q_after_Fading_Rayleigh = Fading_Rayleigh(Distance, I_afterTX, Q_afterTX)
        
        I_afterNoise, Q_afterNoise = ThermalNoise(I_after_Fading_Rayleigh, Q_after_Fading_Rayleigh, SNR)
        
        if Adaptive_Modulation:
            Status, H, P, CRC_status = CovertRX(I_afterNoise, Q_afterNoise)
        else:
            Status, H, P, CRC_status = __CovertRX(I_afterNoise, Q_afterNoise)
        
        if Status:
            BER_sum += 0.5*(XOR_LIST(list(H), header)/max(header_len, len(H)) + XOR_LIST(list(P), MESSAGES[MSG][:payload_len])/max(payload_len, len(P)))
        else:
            Fail+=1
            BER_sum += 0 #1
        
    print('Bit Error Rate = {} for SNR = {}, modulation index = {}, fraction of messages demodulated sucessfully = {} Distance = {}'.format(BER_sum/(Number_of_Messages-Fail+1), SNR, modulation_index, 1-(Fail/Number_of_Messages), Distance))
    file = open("Results_Distance.txt", 'a')
    file.write('Bit Error Rate = {} for SNR = {}, modulation index = {}, fraction of messages demodulated sucessfully = {} Distance = {} \n\n'.format(BER_sum/(Number_of_Messages-Fail+1), SNR, modulation_index, 1-(Fail/Number_of_Messages), Distance))
    file.close() 
    file = open("Results_CSV.txt", 'a')
    file.write('{},{},{},{},{}\n'.format(BER_sum/(Number_of_Messages-Fail+1), SNR, modulation_index, 1-(Fail/Number_of_Messages), Distance))
    file.close() 