# -*- coding: utf-8 -*-

import csv
import sys
import time
import random
import copy
import math
import os
import os.path as pt
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import skimage.io as skimage
from skimage import transform as tr
import skimage.morphology as mor
from argparse import ArgumentParser
from lib.utils import Utils
plt.style.use('ggplot')


class GraphDraw():

    def __init__(self, opbase, roi):
        self.opbase = opbase
        self.roi = roi
        self.scale = 0.8 * 0.8 * 1.75
        self.psep = '/'
        self.x = 127
        self.y = 124
        if roi != 0:
            with open('../GT/10minGroundTruth/CSVfile/test{}.csv'.format(roi), 'r') as f:
                dataReader = csv.reader(f)
                l = [i for i in dataReader]
                self.GTCount = []
                tp = 1
                for i in range(len(l)):
                    if l[i][3] == str(tp):
                        tp += 1
                        self.GTCount.append(0)
                    self.GTCount[tp-2] += 1
        else:
            self.GTCount = None

            
    def graph_draw_number(self, Time, Count):
        # Count
        plt.figure()
        plt.plot(Time, Count)
        if self.GTCount is not None:
            plt.plot(Time, self.GTCount)
            plt.legend(["QCA Net", "Ground Truth"],loc=2)
            ytick = [i for i in range(0, ((np.max(self.GTCount) / 5) + 1) * 5, 5)]
        else:
            plt.legend(["QCA Net"],loc=2)
            ytick = [i for i in range(0, ((np.max(Count) / 5) + 1) * 5, 5)]
        plt.xlabel('Time [day]', size=12)
        plt.ylabel('Number of Nuclei', size=12)
        if Time[-1] != 0:
            plt.xlim([0.0, round(Time[-1], 1)])
        plt.yticks(ytick)
        filename = self.opbase + self.psep + 'Count.pdf'
        plt.savefig(filename)

        
    def graph_draw_volume(self, Time, SumVol, MeanVol, StdVol):
        SumVol = np.array(SumVol) * self.scale
        MeanVol = np.array(MeanVol) * self.scale
        StdVol = np.array(StdVol) * self.scale

        # Volume Mean & SD
        plt.figure()
        plt.plot(Time, MeanVol, color='blue')
        plt.fill_between(Time, np.array(MeanVol)-np.array(StdVol), np.array(MeanVol)+np.array(StdVol), color='blue', alpha=0.4)
        plt.legend(["Mean", "Std. Dev."],loc=1)
        plt.xlabel('Time [day]', size=12)
        plt.ylabel('Volume [$\mu m^{3}$]', size=12)
        if Time[-1] != 0:
            plt.xlim([0.0, round(Time[-1], 1)])
        plt.ylim([0.0, np.max(np.array(MeanVol)+np.array(StdVol)) + 1000])
        filename = self.opbase + self.psep + 'MeanStdVolume.pdf'
        plt.savefig(filename)

        
    def graph_draw_surface(self, Time, SumArea, MeanArea, StdArea):
        SumArea = np.array(SumArea) * self.scale
        MeanArea = np.array(MeanArea) * self.scale
        StdArea = np.array(StdArea) * self.scale

        # Surface Mean & SD
        plt.figure()
        plt.plot(Time, MeanArea, color='blue')
        plt.fill_between(Time, np.array(MeanArea)-np.array(StdArea), np.array(MeanArea)+np.array(StdArea), color='blue', alpha=0.4)
        plt.legend(["Mean", "Std. Dev."],loc=1)
        plt.xlabel('Time [day]', size=12)
        plt.ylabel('Surface Area [$\mu m^{2}$]', size=12)
        if Time[-1] != 0:
            plt.xlim([0.0, round(Time[-1], 1)])
        plt.ylim([0.0, np.max(np.array(MeanArea)+np.array(StdArea)) + 1000])
        filename = self.opbase + self.psep + 'MeanStdSurface.pdf'
        plt.savefig(filename)

        
    def graph_draw_centroid(self, cent_x, cent_y, cent_z):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        #ax = Axes3D(fig)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_xlim(0, self.x)
        ax.set_ylim(0, self.y)
        ax.set_zlim(0, 51)
        cmap =  plt.get_cmap('jet')
        for i in range(len(cent_x)):
            colors = cmap(i / float(len(cent_x)))
            ax.plot(np.array(cent_x[i]), np.array(cent_y[i]), np.array(cent_z[i]), "o", color=colors, alpha=0.6, ms=3, mew=0.5)
        filename = self.opbase + self.psep + 'Centroid.pdf'
        plt.savefig(filename)

        
    def graph_draw_centroid_2axis(self, cent_x, cent_y, axis):
        plt.figure()
        if axis is 'XY':
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.xlim([0, self.x])
            plt.ylim([0, self.y])
        elif axis is 'YZ':
            plt.xlabel('Z')
            plt.ylabel('Y')
            plt.xlim([0, 51])
            plt.ylim([0, self.y])
        elif axis is 'ZX':
            plt.xlabel('X')
            plt.ylabel('Z')
            plt.xlim([0, self.x])
            plt.ylim([0, 51])
        cmap =  plt.get_cmap('jet')
        for i in range(len(cent_x)):
            colors = cmap(i / float(len(cent_x)))
            plt.plot(np.array(cent_x[i]), np.array(cent_y[i]), "o", color=colors, alpha=0.6, ms=3, mew=0.5)
        if axis is 'XY':
            filename = self.opbase + self.psep + 'Centroid-XY.pdf'
        elif axis is 'YZ':
            filename = self.opbase + self.psep + 'Centroid-YZ.pdf'
        elif axis is 'ZX':
            filename = self.opbase + self.psep + 'Centroid-ZX.pdf'
        plt.savefig(filename)
        

        
if __name__ == '__main__':
    ap = ArgumentParser(description='python graph_draw.py')
    ap.add_argument('--input', '-i', nargs='?', default='criteria.csv', help='Specify input files (format : csv)')
    ap.add_argument('--outdir', '-o', nargs='?', default='extract_figs', help='Specify output files directory for create figures')
    ap.add_argument('--roi', '-r', type=int, default=0, help='Specify ROI GT')
    args = ap.parse_args()
    argvs = sys.argv
    util = Utils()
    psep = '/'
    opbase = util.createOpbase(args.outdir)

    # each criterion
    cnt = []
    SumVol, SumArea, Count = [], [], []
    MeanVol, MeanArea, VarVol, VarArea = [], [], [], []
    Cent_X, Cent_Y, Cent_Z = [], [], []

    # import csv
    f = open(args.input, 'r')
    data = csv.reader(f)
    l = []
    for i in data:
        l.append(i)
    f.close()
    l.pop(0)

    for c in range(len(l)):
        if int(l[c][1]) > 0:
            Count.append(int(l[c][1]))
            SumVol.append(float(l[c][2]))
            MeanVol.append(float(l[c][3]))
            VarVol.append(float(l[c][4]))
            SumArea.append(float(l[c][5]))
            MeanArea.append(float(l[c][6]))
            VarArea.append(float(l[c][7]))
            x, y, z = [], [], []
            for i in range(len(l[c][8][1:-1].split(','))):
                x.append(float(l[c][8][1:-1].split(',')[i]))
                y.append(float(l[c][9][1:-1].split(',')[i]))
                z.append(float(l[c][10][1:-1].split(',')[i]))
            Cent_X.append(x)
            Cent_Y.append(y)
            Cent_Z.append(z)
        else:
            Count.append(0)
            SumVol.append(0)
            MeanVol.append(0)
            VarVol.append(0)
            SumArea.append(0)
            MeanArea.append(0)
            VarArea.append(0)
            Cent_X.append([0])
            Cent_Y.append([0])
            Cent_Z.append([0])

    # Time Scale
    dt = 10 / float(60 * 24)
    Time = [dt*x for x in range(len(Count))]

    gd = GraphDraw(opbase, args.roi)
    gd.graph_draw_number(Time, Count)
    gd.graph_draw_volume(Time, SumVol, MeanVol, VarVol)
    gd.graph_draw_surface(Time, SumArea, MeanArea, VarArea)
    gd.graph_draw_centroid(Cent_X, Cent_Y, Cent_Z)
    gd.graph_draw_centroid_2axis(Cent_X, Cent_Y, 'XY')
    gd.graph_draw_centroid_2axis(Cent_Z, Cent_Y, 'YZ')
    gd.graph_draw_centroid_2axis(Cent_X, Cent_Z, 'ZX')
    plt.show()
