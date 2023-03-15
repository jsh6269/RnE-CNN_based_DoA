from matplotlib import pyplot as plt
def compare(name1, name2, cap1, cap2, title, val):
    from loss_plot import interpret
    dict = {'ang':[[-5, 100, -2, 40], 'angle error', '/ang_error.txt'],'loss':[[-5, 100, -0.1, 2.5], 'loss', '/loss.txt'], 'acc':[[-5, 100, 0, 1.2], 'accuracy', '/accuracy.txt']}
    x1, y1 = interpret('./result_final/'+name1+dict[val][2])
    x2, y2 = interpret('./result_final/'+name2+dict[val][2])
    plt.figure(figsize=(6, 6))
    plt.title(title, fontsize = 18)
    plt.plot(x1, y1, label = cap1)
    plt.plot(x2, y2, label = cap2)
    plt.legend([cap1, cap2], fontsize = 14, loc = 'lower right')
    plt.xlabel('epoch', fontsize = 12)
    plt.ylabel(dict[val][1], fontsize = 12)
    plt.axis(dict[val][0])
    import os
    if not os.path.exists('./compare_'+val):
        os.makedirs('./compare_'+val)
    plt.savefig('./compare_' +val+'/'+ cap1+' vs '+cap2+' ' + title+'.png', dpi=100)


def compare2(name1, name2, cap1, cap2, title, val1, val2):
    from loss_plot import interpret
    dict = {'ang':[[-2, 40], 'angle error', '/ang_error.txt'],'loss':[[-0.1, 2.5], 'loss', '/loss.txt'], 'acc':[[0, 1.1], 'accuracy', '/accuracy.txt']}
    fig, ax1 = plt.subplots()
    fig.set_figheight(6)
    fig.set_figwidth(6)
    fig.suptitle(title, fontsize = 16)
    # fig.title = title
    # fig.title_fontsize = 18
    # fig.legend([cap1, cap2], fontsize = 14, loc = 'lower right')
    ax2 = ax1.twinx()
    x1, y1 = interpret('./result_final/'+name1+dict[val1][2])
    ax1.plot(x1, y1)
    x2, y2 = interpret('./result_final/'+name2+dict[val1][2])
    ax1.plot(x2, y2)
    ax1.legend((cap1+' loss', cap2+' loss'), fontsize = 10, loc = 'lower right')

    x3, y3 = interpret('./result_final/' + name1 + dict[val2][2])
    ax2.plot(x3, y3, 'orangered')
    x4, y4 = interpret('./result_final/' + name2 + dict[val2][2])
    ax2.plot(x4, y4, 'darkblue')
    ax2.legend((cap1+' accuracy', cap2+' accuracy'), fontsize = 10, loc = 'upper right')

    ax1.set_xlabel('epoch', fontsize = 12)
    ax1.set_ylabel(dict[val1][1], fontsize = 12)
    ax2.set_ylabel(dict[val2][1], fontsize = 12)
    ax1.set_xlim([-5, 100])
    ax1.set_ylim(dict[val1][0])
    ax2.set_ylim(dict[val2][0])
    import os
    if not os.path.exists('./compare_'+val1+'&'+val2):
        os.makedirs('./compare_'+val1+'&'+val2)
    plt.savefig('./compare_' +val1+'&'+val2+'/'+ cap1+' vs '+cap2+' ' + title+'.png', dpi=100)


for val in [['loss', 'acc']]:
    compare2('conv1d_OriginalAudio1', 'conv2d_OriginalAudio1', 'Conv1d', 'Conv2d', 'Clean_Original Dataset', val[0], val[1])
    compare2('conv1d_OriginalAudio2', 'conv2d_OriginalAudio2', 'Conv1d', 'Conv2d', 'Clean_Augmented Dataset', val[0], val[1])
    compare2('conv1d_Noise1Audio1', 'conv2d_Noise1Audio1', 'Conv1d', 'Conv2d', 'Noise_Original Dataset', val[0], val[1])
    compare2('conv1d_Noise1Audio2', 'conv2d_Noise1Audio2', 'Conv1d', 'Conv2d', 'Noise_Augmented Dataset', val[0], val[1])

    compare2('conv1d_OriginalAudio1', 'conv1d_OriginalAudio2', 'Original', 'Augmented', 'Clean_Conv1d Dataset', val[0], val[1])
    compare2('conv1d_Noise1Audio1', 'conv1d_Noise1Audio2', 'Original', 'Augmented', 'Noise_Conv1d Dataset', val[0], val[1])
    compare2('conv2d_OriginalAudio1', 'conv2d_OriginalAudio2', 'Original', 'Augmented', 'Clean_Conv2d Dataset', val[0], val[1])
    compare2('conv2d_Noise1Audio1', 'conv2d_Noise1Audio2', 'Original', 'Augmented', 'Noise_Conv2d Dataset', val[0], val[1])
