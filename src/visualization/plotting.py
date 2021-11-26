import matplotlib.pyplot as plt

def plot_imgs(images, names, labels, graph_shape = (5, 4)):
    fig, axs = plt.subplots(graph_shape[0], graph_shape[1])
    
    count = 0
    for row in range(graph_shape[0]):
        for column in range(graph_shape[1]):
            img, name, label = images[count], names[count], labels[count]

            axs[row, column].imshow(img)
            axs[row, column].set_title(f"{name} | {label}", fontsize=10)
            count+=1
     
    fig.set_size_inches(24, 24)
    plt.show()
