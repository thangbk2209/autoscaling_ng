import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams["font.family"] = "Arial"


def plot_ipfs():
    y_arr = [0.272468328, 0.376533508, 0.704652309, 5.30307436]
    x_arr = ['100 KB', '1 MB', '10 MB', '100 MB']

    plt.plot(x_arr, y_arr, marker='o')
    plt.xlabel('Data size')
    plt.ylabel('Time (s)')

    # plt.show()
    plt.savefig('arial_ipfs.png')
    plt.savefig('arial_ipfs.pdf')


AES = [0.0011239051818847656, 0.003614187240600586, 0.029785871505737305, 0.2870469093322754]

ECC = [0.0834808349609375, 0.08761715888977051, 0.11416482925415039, 0.37955498695373535]

ABE = [0.044435977935791016, 0.06444096565246582, 0.2779402732849121, 2.591182231903076]

MAABE = [0.04776501655578613, 0.06754398345947266, 0.2799680233001709, 2.521691083908081]


def bar_chart():
    x = [0]
    width = 0.1
    fig, ax = plt.subplots()
    # fig.suptitle("Experiments show the number of nodes on the network becoming a leader", fontsize=11, y=0.95)
    
    # AES
    rects1 = ax.bar(0.1, AES[0], width, color='tab:red', label='100 KB')
    ax.text(0.06, 0.02, str(round(AES[0], 3)), color='blue', Size=5)

    rects1 = ax.bar(0.2, AES[1], width, color='tab:green', label='1 MB')
    ax.text(0.16, 0.02, str(round(AES[1], 3)), color='blue', Size=5)

    rects1 = ax.bar(0.3, AES[2], width, color='tab:orange', label='10 MB')
    ax.text(0.26, AES[2] + 0.02, str(round(AES[2], 3)), color='blue', Size=5)

    rects1 = ax.bar(0.4, AES[3], width, color='tab:blue', label='100 MB')
    ax.text(0.36, AES[3] + 0.02, str(round(AES[3], 3)), color='blue', Size=5)

    # ECC
    rects1 = ax.bar(0.6, ECC[0], width, color='tab:red')
    ax.text(0.56, ECC[0] + 0.02, str(round(ECC[0], 3)), color='blue', Size=5)

    rects1 = ax.bar(0.7, ECC[1], width, color='tab:green')
    ax.text(0.66, ECC[1] + 0.02, str(round(ECC[1], 3)), color='blue', Size=5)

    rects1 = ax.bar(0.8, ECC[2], width, color='tab:orange')
    ax.text(0.76, ECC[2] + 0.02, str(round(ECC[2], 3)), color='blue', Size=5)

    rects1 = ax.bar(0.9, ECC[3], width, color='tab:blue')
    ax.text(0.86, ECC[3] + 0.02, str(round(ECC[3], 3)), color='blue', Size=5)

    # ABE
    rects1 = ax.bar(1.1, ABE[0], width, color='tab:red')
    ax.text(1.06, ABE[0] + 0.02, str(round(ABE[0], 3)), color='blue', Size=5)

    rects1 = ax.bar(1.2, ABE[1], width, color='tab:green')
    ax.text(1.16, ABE[1] + 0.02, str(round(ABE[1], 3)), color='blue', Size=5)

    rects1 = ax.bar(1.3, ABE[2], width, color='tab:orange')
    ax.text(1.26, ABE[2] + 0.02, str(round(ABE[2], 3)), color='blue', Size=5)

    rects1 = ax.bar(1.4, ABE[3], width, color='tab:blue')
    ax.text(1.36, ABE[3] + 0.02, str(round(ABE[3], 3)), color='blue', Size=5)

    # MAABE
    rects1 = ax.bar(1.6, MAABE[0], width, color='tab:red')
    ax.text(1.56, MAABE[0] + 0.02, str(round(MAABE[0], 3)), color='blue', Size=5)

    rects1 = ax.bar(1.7, MAABE[1], width, color='tab:green')
    ax.text(1.66, MAABE[1] + 0.02, str(round(MAABE[1], 3)), color='blue', Size=5)

    rects1 = ax.bar(1.8, MAABE[2], width, color='tab:orange')
    ax.text(1.76, MAABE[2] + 0.02, str(round(MAABE[2], 3)), color='blue', Size=5)

    rects1 = ax.bar(1.9, MAABE[3], width, color='tab:blue')
    ax.text(1.86, MAABE[3] + 0.02, str(round(MAABE[3], 3)), color='blue', Size=5)

    plt.legend()
    # ax.set_xticks(x)
    ax.set_xticklabels(['', '', 'AES', '', 'ECC', '', 'ABE', '', 'MAABE', ''])
    plt.ylim(-0.2, 2.7)
    plt.xlabel("Encryption algorithms")
    plt.ylabel('Time (s)')
    # plt.show()
    plt.savefig('arial_single_encrypt_evaluation.pdf')
    plt.savefig('arial_single_encrypt_evaluation.png')


AECC = [0.0434267520904541, 0.0458219051361084, 0.07522797584533691, 0.33982205390930176]
AABE = [0.04352402687072754, 0.045915842056274414, 0.07182097434997559, 0.32541680335998535]
AMAABE = [0.0481419563293457, 0.04947090148925781, 0.07748293876647949, 0.35279202461242676]


def bar_chart_multiple_encrypt():
    x = [0]
    width = 0.1
    fig, ax = plt.subplots()
    # fig.suptitle("Experiments show the number of nodes on the network becoming a leader", fontsize=11, y=0.95)

    # ECC
    rects1 = ax.bar(0.05, AECC[0], width, color='tab:red', label='100 KB')
    ax.text(0.01, AECC[0] + 0.005, str(round(AECC[0], 3)), color='blue', Size=5)

    rects1 = ax.bar(0.15, AECC[1], width, color='tab:green', label='1 MB')
    ax.text(0.11, AECC[1] + 0.005, str(round(AECC[1], 3)), color='blue', Size=5)

    rects1 = ax.bar(0.25, AECC[2], width, color='tab:orange', label='10 MB')
    ax.text(0.21, AECC[2] + 0.005, str(round(AECC[2], 3)), color='blue', Size=5)

    rects1 = ax.bar(0.35, AECC[3], width, color='tab:blue', label='100 MB')
    ax.text(0.31, AECC[3] + 0.005, str(round(AECC[3], 3)), color='blue', Size=5)

    # ABE
    rects1 = ax.bar(0.65, AABE[0], width, color='tab:red')
    ax.text(0.61, AABE[0] + 0.005, str(round(AABE[0], 3)), color='blue', Size=5)

    rects1 = ax.bar(0.75, AABE[1], width, color='tab:green')
    ax.text(0.71, AABE[1] + 0.005, str(round(AABE[1], 3)), color='blue', Size=5)

    rects1 = ax.bar(0.85, AABE[2], width, color='tab:orange')
    ax.text(0.81, AABE[2] + 0.005, str(round(AABE[2], 3)), color='blue', Size=5)

    rects1 = ax.bar(0.95, AABE[3], width, color='tab:blue')
    ax.text(0.91, AABE[3] + 0.005, str(round(AABE[3], 3)), color='blue', Size=5)

    # MAABE
    rects1 = ax.bar(1.25, AMAABE[0], width, color='tab:red')
    ax.text(1.21, AMAABE[0] + 0.005, str(round(AMAABE[0], 3)), color='blue', Size=5)

    rects1 = ax.bar(1.35, AMAABE[1], width, color='tab:green')
    ax.text(1.31, AMAABE[1] + 0.005, str(round(AMAABE[1], 3)), color='blue', Size=5)

    rects1 = ax.bar(1.45, AMAABE[2], width, color='tab:orange')
    ax.text(1.41, AMAABE[2] + 0.005, str(round(AMAABE[2], 3)), color='blue', Size=5)

    rects1 = ax.bar(1.55, AMAABE[3], width, color='tab:blue')
    ax.text(1.51, AMAABE[3] + 0.005, str(round(AMAABE[3], 3)), color='blue', Size=5)

    plt.legend(loc='upper left')
    # ax.set_xticks(x)
    labels = [item.get_text() for item in ax.get_xticklabels()]
    labels[2] = 'AES-ECC'
    labels[5] = 'AES-ABE'
    labels[8] = 'AES-MAABE'
    # print(labels)
    ax.set_xticklabels(labels)
    plt.xlim(-0.1, 1.7)
    plt.xlabel("Encryption algorithms")
    plt.ylabel('Time (s)')
    # plt.show()
    plt.savefig('arial_multi_encrypt_evaluation.pdf')
    plt.savefig('arial_multi_encrypt_evaluation.png')


if __name__ == "__main__":
    # bar_chart()
    # bar_chart_multiple_encrypt()
    # plot_ipfs()