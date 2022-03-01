'''
Author: your name
Date: 2022-03-01 23:41:53
LastEditTime: 2022-03-01 23:41:54
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /unofficial_HGCNplus/utils/drawer.py
'''
import matplotlib.pyplot as plt
import numpy as np
import os


def plot_mesh(img, title="", save_home=""):
    img = img
    fig, ax = plt.subplots()
    plt.title(title)
    fig.colorbar(plt.pcolormesh(range(img.shape[1]), range(img.shape[0]), img))
    if save_home != "":
        print(os.path.join(save_home, "%s.jpg" % title))
        plt.savefig(os.path.join(save_home, "%s.jpg" % title))
        return
    # plt.show()


def plot_spec_mesh(img, title="", save_home=""):
    img = np.log(abs(img))
    fig, ax = plt.subplots()
    plt.title(title)
    fig.colorbar(plt.pcolormesh(range(img.shape[1]), range(img.shape[0]), img))
    if save_home != "":
        print(os.path.join(save_home, "%s.jpg" % title))
        plt.savefig(os.path.join(save_home, "%s.jpg" % title))
    plt.show()


def plot_scatter(array, title="1"):
    xs = np.arange(len(array))
    plt.scatter(xs, array)
    plt.title(title)
    plt.show()


def plot(array, title):
    plt.plot(array)
    plt.title(title)
    plt.show()
