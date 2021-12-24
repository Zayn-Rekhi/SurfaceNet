import matplotlib.pyplot as plt

def plot_imgs(images, names, labels, graph_shape = (5, 4)):
    fig, axs = plt.subplots(graph_shape[0], graph_shape[1])
    
    count = 0
    for row in range(graph_shape[0]):
        for column in range(graph_shape[1]):
            img, name, label = images[count], names[count], labels[count]
            
            axs[row, column].imshow(img, cmap='gray', vmin=0, vmax=1)
            axs[row, column].set_title(f"{label}", fontsize=10)
            axs[row, column].set_xlabel(f"{name}")
            count+=1
     
    fig.set_size_inches(24, 24)
    plt.show()
