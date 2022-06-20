from matplotlib import pyplot as plt


def FilePath(): #choose Files with file explorer pop up and return num of files and paths
    import tkinter as tk
    from tkinter import filedialog
    import os
    application_window = tk.Tk()
    my_filetypes = [('all files', '.*'), ('text files', '.txt')]
    answer = filedialog.askopenfilenames(parent=application_window,
                                     initialdir=os.getcwd(),
                                     title="Please select one or more files:",
                                     filetypes=my_filetypes)
    
    N_files = len(answer) 
    print('Now,b get ' + str(N_files) + ' files')
    print(answer)
    application_window.destroy()
    return  answer # the answer was a tuple object, pay attentions!!!


def DMReader(filepaths):
    import ncempy.io as nio
    dataset = []
    pixelSizes = []
    for i in filepaths:
        data = nio.read(i)
        img = data['data']
        pixsize = data['pixelSize']
        dataset.append(img)
        pixelSizes.append(pixsize)
    print(f'We got {len(dataset)} images here.')
    return dataset, pixelSizes


def plot_labeled_image(image, centroids, cmap='gray', peaks=None, ax=None, alpha=.8, extent=[(0, 1024), (0, 1024)]):
    if peaks is not None:
        centroids = peaks

    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(centroids[:, 0], centroids[:, 1], 'o',
            ms=.5, mec='m', mfc='g', alpha=alpha)
    ax.imshow(image.T, cmap=cmap)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(extent[0])
    ax.set_ylim(extent[1])
    ax.set_aspect(1)
 
    

