#Inisialisasi package & Library
import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext
from tkinter import Menu
from tkinter import messagebox as msg
from tkinter import Spinbox
from tkinter import filedialog as fd
from tkinter.filedialog import askopenfilename
from os import path
from tkinter import *
import csv
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD 
from sklearn.decomposition import PCA
import nltk
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import average, fcluster
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics.pairwise import cosine_distances
from nltk.tokenize import WordPunctTokenizer
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup
from PIL import Image, ImageTk

#open directory file csv
def read_csv():
   global fileopen
    
   filename=fd.askopenfilename()
   Entry_name_open.delete(0, END)
   Entry_name_open.insert(0, filename)
   nama_file1 = Entry_name_open.get()
   file=nama_file1.split('/')
   fileopen=str(file[-1])
   print(fileopen)

#import file csv
def import_csv_data():
    global vektor
    global vektor1
    global listAgregat
    global terms
    global status
    global df
    global list_stopword
    global list_data
    
    with open(fileopen) as f:
        ulasan=f.readlines()
    
    df = pd.DataFrame(ulasan,columns=['text']) #load data ke python

    listAgregat = [] #inisiasi list kosong
    for i in df.text:
        listAgregat.append(i)
    print(listAgregat)
    vectorizer = TfidfVectorizer(max_features= 1000, norm='l2', # keep top 1000 terms 
                                 use_idf=True, min_df=2, 
                                 smooth_idf=False)

    vektor = vectorizer.fit_transform(listAgregat)
    print(vektor)
     #perhitungan TF-IDF
    vektor1 = vektor.toarray()
    terms = vectorizer.get_feature_names() #pengambilan kata bersesuaian
    messagebox.showinfo("Message","Processing dan Pembobotan Berhasil")

    
def cekTruncated():#cek nilai truncated svd yg sesuai
    global V
    global svdfix
    
    u,s,vh=np.linalg.svd(vektor1, full_matrices=True)
    sum_s=np.sum(s)
    cumsum_s=np.cumsum(s)
    plot_var=cumsum_s/sum_s
    for i,v in enumerate (plot_var):
        if v>=0.8:
            trunc=i
            break 
        
     #membuat grafik 
    figure1 = plt.Figure(figsize=(5,4), dpi=80) 
    ax = figure1.add_subplot(111)
    ax.plot(plot_var)
    line = FigureCanvasTkAgg(figure1, win)
    ax.set_xlabel('Banyaknya PC')
    ax.set_ylabel('Nilai')
    ax.set_title('Plot Pemilihan Nilai K')
    ax.axhline(linewidth=2, color='r', y=0.8)
    line.get_tk_widget().grid(row=32, column=0, columnspan=11)
    messagebox.showinfo("Message","Plot pemilihan nilai k sukses")
    
#proses trtuncated svd  
def truncatedSVD():
    global svdfix
    
    trunc=int(entry_banyaktrunc.get())
    if ((trunc <1) or (trunc >= vektor.shape[1])):
        messagebox.showerror('Error', 'Angka yang Anda Masukkan Tidak Valid')
        entry_banyaktrunc.delete(0,END)
    else:
        svd=TruncatedSVD(n_components=trunc, algorithm='randomized')
        svdfix=svd.fit_transform(vektor1)
        messagebox.showinfo("Message","Truncated SVD Berhasil")


def clustering():
    global jumlah
    global cluster
    global clust
    global sortClust
    global Z
    
    n_clust=float(entry_banyakclust.get())
    Y=pdist(svdfix, 'cosine')
    Z=linkage(Y,'average')
        
    cluster= fcluster(Z, t=n_clust, criterion='distance')
    jumlah=np.max(cluster)
    ms = list(zip(cluster, listAgregat)) 
    clust = pd.DataFrame(ms,columns = ['Clusters', 'ulasan']) 
    sortClust=clust.reindex(clust.Clusters.astype(int).sort_values().index) 
    print(sortClust)
    plt.figure(figsize=(5,4))
    plt.title('HCA Dendrogram')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    dendrogram(Z,leaf_rotation=90,leaf_font_size=12,)
    plt.savefig('dendrogram.png')
    image = plt.imread('dendrogram.png')
    fig = plt.figure(figsize=(5,4), dpi=80)
    im = plt.imshow(image) # later use a.set_data(new_data)
    ax = fig.add_subplot(111)
    ax.set_xticklabels([]) 
    ax.set_yticklabels([]) 
    
    # a tk.DrawingArea
    canvas = FigureCanvasTkAgg(fig, win)
    canvas.show()
    canvas.get_tk_widget().grid(row=32, column=0, columnspan=11)

    
def tombol_detail_database():
    database = Toplevel(win)
    database.geometry('700x400')
    database.configure(bg='white')
    database.title('Details Database')
    
    list1 = []
    for kata in listAgregat:
        list1.append(kata)
            
        txt = Text(database)
        txt.grid(row=0, column=0, sticky="eswn")
        
        
        scroll_y = Scrollbar(database, orient="vertical", command=txt.yview)
        scroll_y.grid(row=0, column=1, sticky="ns")
      
        txt.configure(yscrollcommand=scroll_y.set)
        
        very_long_list = "\n".join([str(i) for i in list1])
        
        txt.insert("1.0", very_long_list)

        txt.configure(state="disabled", relief="flat", bg=database.cget("bg"))

def tombol_detail_cluster():
    cluster=Toplevel(win)
    cluster.geometry('800x500')
    cluster.configure(bg='white')
    cluster.title('Detail Hasil Cluster')  
    
    tes1 = sortClust['Clusters'].tolist()
    tes2 = sortClust['ulasan'].tolist()
    tes3 = list(zip(tes1, tes2))
    
    txt = Text(cluster)
    txt.grid(row=0, column=0, sticky="eswn")
    
    
    scroll_y = Scrollbar(cluster, orient="vertical", command=txt.yview)
    scroll_y.grid(row=0, column=1, sticky="ns")
  
    txt.configure(yscrollcommand=scroll_y.set)
    
    very_long_list = "\n".join([str(i) for i in tes3])
    
    txt.insert("1.0", very_long_list)
    
    
    txt.configure(state="disabled", relief="flat", bg=cluster.cget("bg"))

def keyword_cluster():
    jumlah_cluster= max(sortClust['Clusters'])
    print(jumlah_cluster)
    kumpul=[]
    for i in range (1,jumlah_cluster+1):
        temp=sortClust[sortClust['Clusters']==i]
        kalimat_clust1=temp['ulasan'].tolist()
        vectorizer1=TfidfVectorizer(min_df=1, norm='l1', ngram_range=(1,2))
        c1=vectorizer1.fit_transform(kalimat_clust1)
        df_c1=pd.DataFrame(c1.toarray(),columns=vectorizer1.get_feature_names())
        weight_cq={}
        for i in df_c1.columns:
            weight_cq[i]=sum(df_c1.loc[:,i])
        df_bobot=pd.DataFrame(weight_cq, index=['bobot']).transpose()
        urut=df_bobot.sort_values('bobot', ascending=False).head(3)
        kumpul.append(urut)
    
    print(kumpul)
    
    cluster=Toplevel(win)
    cluster.geometry('800x500')
    cluster.configure(bg='white')
    cluster.title('Detail Hasil Cluster')  
    
    
    txt = Text(cluster)
    txt.grid(row=0, column=0, sticky="eswn")
    
    
    scroll_y = Scrollbar(cluster, orient="vertical", command=txt.yview)
    scroll_y.grid(row=0, column=1, sticky="ns")
  
    txt.configure(yscrollcommand=scroll_y.set)
    
    very_long_list = "\n".join([str(i) for i in kumpul])
    
    txt.insert("1.0", very_long_list)
    
    
    txt.configure(state="disabled", relief="flat", bg=cluster.cget("bg"))
                
def tombolclear():
    Entry_name_open.config(state=NORMAL)
    Entry_name_open.delete(0, END)
    tombol_load.config(state=NORMAL)    
    tombol_proces.config(state=NORMAL)
    
    entry_banyaktrunc.config(state=NORMAL)
    entry_banyaktrunc.delete(0, END)
    tombol_cek_trun.config(state=NORMAL)
    tombol_trunc.config(state=NORMAL)
    
    entry_banyakclust.config(state=NORMAL)
    entry_banyakclust.delete(0, END)
    tombol_sc.config(state=NORMAL)
    tombol_database.config(state=NORMAL)
    tombol_clustering.config(state=NORMAL)
    
win=tk.Tk()
win.geometry('725x700')
win.configure(bg="tan")
win.columnconfigure(20, minsize=10)
win.rowconfigure(18,minsize=10)

win.title("Tugas Akhir")
lblInfo1 = Label(font=('calibri', 12),justify='center', text = 'Implementasi Text Mining untuk Pengelompokan Ulasan Pelanggan E-Commerce Berdasarkan Topik Ulasan\n Oleh :  Li''Izza Diana Manzil NRP. 06111540000049 \n Supervisor : Prof. Dr. Mohammad Isa Irawan, M.T. ', 
                fg='black',bg="bisque", bd=15)
lblInfo1.grid(row=0, columnspan=20)

win.resizable(False, True)

#BAGAIN KIRI
#OPEN FILE

lbl_open = Label(font=('calibri', 10,'bold'), text = 'Open File', 
                fg='black',bg='tan', bd=7)
lbl_open.grid(row=5, column=0)

Entry_name_open = Entry(win, font=('calibri',10,'bold'), justify='center', width=20)
Entry_name_open.grid(row=6, column=0)

tombol_load = Button(padx=5, pady=0.1, fg="black", font=('arial', 8, 'bold'), text='Open', bg='dark salmon',
             command=read_csv)
tombol_load.grid(row=6, column=1)


tombol_proces=Button(padx=5, pady=0.1, fg="black", font=('arial', 8, 'bold'), text='Process', bg='dark salmon',
             command=import_csv_data)
tombol_proces.grid(row=7, column=0)


#Truncated Parameter

lbl_truncated = Label(font=('calibri', 10, 'bold'), text = 'Truncated SVD', 
                fg='black',bg='tan',bd=7)
lbl_truncated.grid(row=13, column=0)

tombol_cek_trun = Button(padx=5, pady=1, fg="black", font=('arial', 8, 'bold'), text='Cek Nilai Truncated', bg='dark salmon',
             command=cekTruncated)
tombol_cek_trun.grid(row=16,column=0, columnspan=1)

entry_banyaktrunc=Entry(win, font=('calibri',10,'bold'), justify='center', width=10)
entry_banyaktrunc.grid(row=17, column=0)
lbl_truncated = Label(font=('calibri', 10, 'bold'), text = 'Nilai Truncated SVD', 
                fg='black',bg='tan',bd=7)
lbl_truncated.grid(row=17, column=1)

tombol_trunc = Button(padx=5, pady=1, fg="black", font=('arial', 10, 'bold'), text='Truncated', bg='dark salmon',
             command=truncatedSVD)
tombol_trunc.grid(row=18,column=0, columnspan=1)

lbl_banyaktruncated = Label(font=('calibri', 10, 'bold'), text = 'Clustering', 
                fg='black',bg='tan',bd=7)
lbl_banyaktruncated.grid(row=5, column=3)


lbl_threshold = Label(font=('calibri', 10, 'bold'), text = 'Threshold', 
                fg='black',bg='tan',bd=7)
lbl_threshold.grid(row=6, column=4)

entry_banyakclust=Entry(win, font=('calibri',10,'bold'), justify='center', width=10)
entry_banyakclust.grid(row=6, column=3)

tombol_sc=Button(padx=5, pady=1, fg="black", font=('arial', 10, 'bold'), text='Clustering', bg='dark salmon',
             command=clustering)
tombol_sc.grid(row=7,column=3, columnspan=1)

#tombol detail
lbl_plot = Label(font=('calibri', 10, 'bold'), text = 'Grafik', 
                fg='black',bg='tan',bd=7)
lbl_plot.grid(row=31, column=0)

tombol_database = Button(win, width=10, height=1, fg="black", font=('arial', 10, 'bold'), text='Database', bg='dark salmon',
                         command=tombol_detail_database)
tombol_database.grid(row=30,column=2)

tombol_clustering = Button(win, width=12, height=1, fg="black", font=('arial', 10, 'bold'), text='Hasil Cluster', bg='dark salmon',
                         command=tombol_detail_cluster)
tombol_clustering.grid(row=30,column=3)

tombol_reset = Button(win, width=10, height=1, fg="black", font=('arial', 10, 'bold'), text='Reset', bg='dark salmon',
                         command=tombolclear)
tombol_reset.grid(row=30,column=4)

tombol_keyword = Button(win, width=12, height=1, fg="black", font=('arial', 10, 'bold'), text='Keyword', bg='dark salmon',
                         command=keyword_cluster)
tombol_keyword.grid(row=31,column=3)

win.mainloop()
