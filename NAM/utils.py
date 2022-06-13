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


def hpFunction():
    f = hp.File('testFilter1.hdf5', 'w')
    dset = f.create_dataset('bkgs', data=img_sbkg)
    f['guassian_filetered']=filtered2
    f.close()
