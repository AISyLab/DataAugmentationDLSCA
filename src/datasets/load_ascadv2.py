import numpy as np
import h5py
from tensorflow.keras.utils import *
from sklearn.preprocessing import StandardScaler
from numba import njit
from scalib.metrics import SNR

aes_sbox = np.array([
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
])

""" G auxiliary function that is used to generate the permution of indices """
G = np.array([0x0C, 0x05, 0x06, 0x0b, 0x09, 0x00, 0x0a, 0x0d, 0x03, 0x0e, 0x0f, 0x08, 0x04, 0x07, 0x01, 0x02])

"""
Two Tables to process a field multiplication over GF(256): a*b = alog (log(a) + log(b) mod 255)
"""
log_table = np.array([0, 0, 25, 1, 50, 2, 26, 198, 75, 199, 27, 104, 51, 238, 223, 3,
                      100, 4, 224, 14, 52, 141, 129, 239, 76, 113, 8, 200, 248, 105, 28, 193,
                      125, 194, 29, 181, 249, 185, 39, 106, 77, 228, 166, 114, 154, 201, 9, 120,
                      101, 47, 138, 5, 33, 15, 225, 36, 18, 240, 130, 69, 53, 147, 218, 142,
                      150, 143, 219, 189, 54, 208, 206, 148, 19, 92, 210, 241, 64, 70, 131, 56,
                      102, 221, 253, 48, 191, 6, 139, 98, 179, 37, 226, 152, 34, 136, 145, 16,
                      126, 110, 72, 195, 163, 182, 30, 66, 58, 107, 40, 84, 250, 133, 61, 186,
                      43, 121, 10, 21, 155, 159, 94, 202, 78, 212, 172, 229, 243, 115, 167, 87,
                      175, 88, 168, 80, 244, 234, 214, 116, 79, 174, 233, 213, 231, 230, 173, 232,
                      44, 215, 117, 122, 235, 22, 11, 245, 89, 203, 95, 176, 156, 169, 81, 160,
                      127, 12, 246, 111, 23, 196, 73, 236, 216, 67, 31, 45, 164, 118, 123, 183,
                      204, 187, 62, 90, 251, 96, 177, 134, 59, 82, 161, 108, 170, 85, 41, 157,
                      151, 178, 135, 144, 97, 190, 220, 252, 188, 149, 207, 205, 55, 63, 91, 209,
                      83, 57, 132, 60, 65, 162, 109, 71, 20, 42, 158, 93, 86, 242, 211, 171,
                      68, 17, 146, 217, 35, 32, 46, 137, 180, 124, 184, 38, 119, 153, 227, 165,
                      103, 74, 237, 222, 197, 49, 254, 24, 13, 99, 140, 128, 192, 247, 112, 7])

alog_table = np.array([1, 3, 5, 15, 17, 51, 85, 255, 26, 46, 114, 150, 161, 248, 19, 53,
                       95, 225, 56, 72, 216, 115, 149, 164, 247, 2, 6, 10, 30, 34, 102, 170,
                       229, 52, 92, 228, 55, 89, 235, 38, 106, 190, 217, 112, 144, 171, 230, 49,
                       83, 245, 4, 12, 20, 60, 68, 204, 79, 209, 104, 184, 211, 110, 178, 205,
                       76, 212, 103, 169, 224, 59, 77, 215, 98, 166, 241, 8, 24, 40, 120, 136,
                       131, 158, 185, 208, 107, 189, 220, 127, 129, 152, 179, 206, 73, 219, 118, 154,
                       181, 196, 87, 249, 16, 48, 80, 240, 11, 29, 39, 105, 187, 214, 97, 163,
                       254, 25, 43, 125, 135, 146, 173, 236, 47, 113, 147, 174, 233, 32, 96, 160,
                       251, 22, 58, 78, 210, 109, 183, 194, 93, 231, 50, 86, 250, 21, 63, 65,
                       195, 94, 226, 61, 71, 201, 64, 192, 91, 237, 44, 116, 156, 191, 218, 117,
                       159, 186, 213, 100, 172, 239, 42, 126, 130, 157, 188, 223, 122, 142, 137, 128,
                       155, 182, 193, 88, 232, 35, 101, 175, 234, 37, 111, 177, 200, 67, 197, 84,
                       252, 31, 33, 99, 165, 244, 7, 9, 27, 45, 119, 153, 176, 203, 70, 202,
                       69, 207, 74, 222, 121, 139, 134, 145, 168, 227, 62, 66, 198, 81, 243, 14,
                       18, 54, 90, 238, 41, 123, 141, 140, 143, 138, 133, 148, 167, 242, 13, 23,
                       57, 75, 221, 124, 132, 151, 162, 253, 28, 36, 108, 180, 199, 82, 246, 1])


@njit
def permIndices(i, m0, m1, m2, m3):
    """
    The permutation function on the 16 indices i. The function is defined from the masks m0, m1, m2, and m3.
    """
    x0, x1, x2, x3 = m0 & 0x0f, m1 & 0x0f, m2 & 0x0f, m3 & 0x0f
    return G[G[G[G[(15 - i) ^ x0] ^ x1] ^ x2] ^ x3]


@njit
def multGF256(a, b):
    """ Multiplication function in GF(2^8) """
    if (a == 0) or (b == 0):
        return 0
    else:
        return alog_table[(log_table[a] + log_table[b]) % 255]


def get_snr_peaks(traces, labels, poi):
    labels_for_snr = [[label] for label in labels]
    snr = SNR(np=1, ns=traces.shape[1], nc=256)
    snr.fit_u(l=np.array(traces, dtype=np.int16), x=np.array(labels_for_snr, dtype=np.uint16))
    snr_val = snr.get_snr()
    peaks = np.sort(np.argsort(snr_val[0])[::-1][:poi])
    return traces[:, peaks], peaks


class ReadASCADv2:

    def __init__(self, n_profiling, n_attack, target_byte, leakage_model, file_path, first_sample=0, number_of_samples=700):
        self.name = "ascadv2"
        self.n_profiling = n_profiling
        self.n_attack = n_attack
        self.target_byte = target_byte
        self.leakage_model = leakage_model
        self.file_path = file_path
        self.fs = first_sample
        self.ns = number_of_samples
        self.classes = 9 if leakage_model == "HW" else 256

        self.x_profiling = []
        self.x_attack = []

        self.y_profiling = []
        self.y_attack = []

        self.profiling_labels = []
        self.attack_labels = []

        self.labels_key_hypothesis_attack = None
        self.share1_profiling, self.share2_profiling, self.share1_attack, self.share2_attack = None, None, None, None

        self.round_key = "00000000000000000000000000000000"
        self.correct_key = bytearray.fromhex(self.round_key)[target_byte]

        self.load_dataset()

    def load_dataset(self):
        in_file = h5py.File(self.file_path, "r")

        attack_plaintext = in_file['Attack_traces/metadata']['plaintext']
        attack_key = in_file['Attack_traces/metadata']['key']
        profiling_mask = in_file['Profiling_traces/metadata']['masks']
        attack_mask = in_file['Attack_traces/metadata']['masks']

        profiling_masks = profiling_mask[:self.n_profiling]
        attack_plaintexts = attack_plaintext[:self.n_attack]
        attack_keys = attack_key[:self.n_attack]
        attack_masks = attack_mask[:self.n_attack]

        label_type = "second_order"
        order = 2
        file_labels_profiling = h5py.File('/tudelft.net/staff-umbrella/dlsca/Guilherme/ascadv2-labels-Profiling_traces.h5', "r")
        file_labels_attack = h5py.File('/tudelft.net/staff-umbrella/dlsca/Guilherme/ascadv2-labels-Attack_traces.h5', "r")
        self.profiling_labels = np.array(file_labels_profiling[f'Profiling_traces/labels_{label_type}'])[:, self.target_byte]
        self.attack_labels = np.array(file_labels_attack[f'Attack_traces/labels_{label_type}'])[:, self.target_byte]

        self.share1_profiling, self.share2_profiling, self.share1_attack, self.share2_attack = self.create_shares(profiling_masks,
                                                                                                                  self.profiling_labels,
                                                                                                                  attack_masks,
                                                                                                                  self.attack_labels)

        if self.leakage_model == "HW":
            self.profiling_labels = [bin(int(iv)).count("1") for iv in self.profiling_labels]
            self.attack_labels = [bin(int(iv)).count("1") for iv in self.attack_labels]

        self.y_profiling = to_categorical(self.profiling_labels, num_classes=self.classes)
        self.y_attack = to_categorical(self.attack_labels, num_classes=self.classes)

        self.labels_key_hypothesis_attack = self.create_labels_key_guess(attack_plaintexts, attack_keys, attack_masks, leakage_order=order)

        if self.ns == 100:
            """ Then it is white-box"""
            profiling_traces_part1, peaks_1 = get_snr_peaks(in_file['Profiling_traces/traces'][:, 5300:5800], self.share1_profiling, 50)
            profiling_traces_part2, peaks_2 = get_snr_peaks(in_file['Profiling_traces/traces'][:, 11250:11750], self.share2_profiling, 50)
            self.x_profiling = np.concatenate((profiling_traces_part1, profiling_traces_part2), axis=1)

            attack_traces_part1 = in_file['Attack_traces/traces'][:, 5300:5800]
            attack_traces_part2 = in_file['Attack_traces/traces'][:, 11250:11750]
            self.x_attack = np.concatenate((attack_traces_part1[:, peaks_1], attack_traces_part2[:, peaks_2]), axis=1)
        else:
            profiling_traces_part1 = in_file['Profiling_traces/traces'][:, 5300:5800]
            profiling_traces_part2 = in_file['Profiling_traces/traces'][:, 11250:11750]
            self.x_profiling = np.concatenate((profiling_traces_part1, profiling_traces_part2), axis=1)

            attack_traces_part1 = in_file['Attack_traces/traces'][:, 5300:5800]
            attack_traces_part2 = in_file['Attack_traces/traces'][:, 11250:11750]
            self.x_attack = np.concatenate((attack_traces_part1, attack_traces_part2), axis=1)

    def rescale(self, reshape_to_cnn):
        scaler = StandardScaler()
        self.x_profiling = scaler.fit_transform(self.x_profiling)
        self.x_attack = scaler.transform(self.x_attack)

        if reshape_to_cnn:
            print("reshaping to 3 dims")
            self.x_profiling = self.x_profiling.reshape((self.x_profiling.shape[0], self.x_profiling.shape[1], 1))
            self.x_attack = self.x_attack.reshape((self.x_attack.shape[0], self.x_attack.shape[1], 1))

    def create_labels_key_guess(self, plaintexts, keys, masks, shuffling=False, leakage_order=3):
        nt = len(plaintexts)
        labels_key_hypothesis = np.zeros((256, nt), dtype=np.int)

        """ remove shuffling countermeasure """
        if shuffling:
            ind = np.full(len(plaintexts), self.target_byte)
        else:
            ind = permIndices(self.target_byte, masks[:, 0], masks[:, 1], masks[:, 2], masks[:, 3])
        plaintext = [p[ind[i]] for i, p in enumerate(plaintexts)]
        actual_key_byte = [k[ind[i]] for i, k in enumerate(keys)]

        for trace_index in range(nt):
            idx_hypothesis = 1
            for key_byte_hypothesis in range(256):
                """ create intermediate for the guessed plaintext-key combination """
                intermediate = aes_sbox[int(plaintext[trace_index]) ^ int(key_byte_hypothesis)]
                if leakage_order == 1:
                    intermediate = multGF256(masks[trace_index, 18], intermediate) ^ masks[trace_index, 17]
                elif leakage_order == 2:
                    intermediate = multGF256(masks[trace_index, 18], intermediate)

                """ check leakage model """
                if self.leakage_model == "HW":
                    intermediate = bin(intermediate).count("1")

                key_hypothesis_index = 0 if key_byte_hypothesis == actual_key_byte[trace_index] else idx_hypothesis
                labels_key_hypothesis[key_hypothesis_index][trace_index] = intermediate
                if key_byte_hypothesis != actual_key_byte[trace_index]:
                    idx_hypothesis += 1

        return labels_key_hypothesis

    def create_shares(self, profiling_masks, profiling_labels, attack_masks, attack_labels):

        share1_profiling = [int(label) ^ int(beta) for label, beta in zip(profiling_labels[:], profiling_masks[:, 17])]
        share2_profiling = [beta for beta in profiling_masks[:, 17]]
        share1_attack = [int(label) ^ int(beta) for label, beta in zip(attack_labels[:], attack_masks[:, 17])]
        share2_attack = [beta for beta in attack_masks[:, 17]]

        if self.leakage_model == "HW":
            share1_profiling = [bin(iv).count("1") for iv in share1_profiling]
            share2_profiling = [bin(iv).count("1") for iv in share2_profiling]
            share1_attack = [bin(iv).count("1") for iv in share1_attack]
            share2_attack = [bin(iv).count("1") for iv in share2_attack]

        return share1_profiling, share2_profiling, share1_attack, share2_attack
