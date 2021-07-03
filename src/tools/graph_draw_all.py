# -*- coding: utf-8 -*-

import csv
import sys
import time
import random
import copy
import math
import os
sys.path.append(os.getcwd())
import os.path as pt
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from glob import glob
import skimage.io as skimage
from skimage import transform as tr
import skimage.morphology as mor
from argparse import ArgumentParser
from src.lib.utils import createOpbase
plt.style.use('ggplot')


class GraphDrawAll():

    def __init__(self, opbase, roi, filename):
        self.opbase = opbase
        self.roi = roi
        #self.scale = 0.8 * 0.8 * 1.75
        self.scale = 0.8 * 0.8 * 2.0
        self.psep = '/'
        self.x = 127
        self.y = 124
        self.z = 111
        self.density = 0
        self.roi_pixel_num = 0
        self.filename = filename
        if roi != 0:
            with open('GT/10minGroundTruth/CSVfile/test{}.csv'.format(roi), 'r') as f:
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
        cmap =  plt.get_cmap('Paired')
        plt.figure()
        for num in range(len(Count)):
            colors = cmap(float(num) / len(Count))
            #if np.unique((np.array(Count[num][:100]) > 8) * 1)[0] == 0 and len(np.unique((np.array(Count[num][:100]) > 8) * 1)) < 2:
            plt.plot(Time[:len(Count[num])], Count[num], color=colors, alpha=0.8, linewidth=1.0)
            #else:
            #    print(self.filename[num])
        # plt.legend(['Emb.1', 'Emb.2', 'Emb.3', 'Emb.4', 'Emb.5', 'Emb.6', 'Emb.7', 'Emb.8', 'Emb.9', 'Emb.10', 'Emb.11'] ,loc=2)
        plt.xlabel('Time [day]', size=12)
        plt.ylabel('Number of Nuclei', size=12)
        if Time[-1] != 0:
            plt.xlim([0.0, round(Time[-1], 1)])
        ytick = [i for i in range(0, int(round(((np.max(Count) / 5) + 1))) * 5, 5)]
        plt.yticks(ytick)
        filename = self.opbase + self.psep + 'Count_all.pdf'
        plt.savefig(filename)


    def graph_draw_synchronicity(self, Time, Count):
        # Count
        #cmap =  plt.get_cmap('Paired')
        #cell_stage = [2, 3, 4, 5, 8, 9, 16, 17, 32]
        cell_stage = [2, 3, 4, 5, 8, 9, 16, 17, 32, 128]
        #label = ['2-cell stage', '3-cell stage', '4-cell stage', '5- to 7-cell stage', '8-cell stage', '9- to 15-cell stage', '16-cell stage', '17- to 31-cell stage', '32-cell stage or more']
        label = ['2-cell stage', '3-cell stage', '4-cell stage', '5- to 7-cell stage', '8-cell stage', '9- to 15-cell stage', '16-cell stage', '17- to 31-cell stage', '32-cell stage', '33- to 63-cell stage', '64-cell stage', '65- to 127-cell stage', '128-cell stage or more']
        all_period_cell = []
        cmap =  plt.get_cmap('Paired')
        #['2-cell stage', '4-cell stage', '8-cell stage', '16-cell stage', '32-cell stage']
        plt.figure()
        for num in range(len(Count)):
            period_cell = []
            current_state = 1
            consist_flag = 0
            for tp in range(len(Count[num])):
                if Count[num][tp] >= cell_stage[current_state]:
                    consist_flag += 1
                    if consist_flag > 5:
                        current_state += 1
                        period_cell.append(Time[tp])
                        consist_flag = 0
                else:
                    consist_flag = 0
            period_cell.append(Time[tp])
            ''' for legend '''
            for i in range(len(period_cell)):
                colors = cmap(i+1)
                plt.barh(len(Count) - num, period_cell[i], height=0.0001, align='center', color=colors)
            ''' for plot '''
            for i in range(len(period_cell)):
                colors = cmap(len(period_cell) - i)
                plt.barh(len(Count) - num, period_cell[-(i+1)], height=0.8, align='center', label=label[len(period_cell) - (i+1)], color=colors)
        plt.yticks(range(1, len(Count)), ["" for i in range(len(Count))])
        plt.xlabel('Time [day]', size=12)
        plt.xlim([0.0, 3.5])
        filename = self.opbase + self.psep + 'CellDivSync.pdf'
        plt.savefig(filename)

        plt.figure()
        for i in range(len(period_cell)):
            colors = cmap(i+1)
            plt.plot(1, 1, color=colors)
        plt.legend(label, loc=1)
        filename = self.opbase + self.psep + 'CellDivSync_legend.pdf'
        plt.savefig(filename)


    def graph_draw_volume(self, Time, SumVol, MeanVol, StdVol):
        SumVol = np.array(SumVol) * self.scale
        MeanVol = np.array(MeanVol) * self.scale
        StdVol = np.array(StdVol) * self.scale

        # Volume Mean & SD
        cmap =  plt.get_cmap('Paired')
        plt.figure()
        for num in range(len(SumVol)):
            colors = cmap(float(num) / len(SumVol))
            plt.plot(Time, MeanVol[num], color=colors, alpha=0.8, linewidth=1.0)
        #plt.legend(['Emb.1', 'Emb.2', 'Emb.3', 'Emb.4', 'Emb.5', 'Emb.6', 'Emb.7', 'Emb.8', 'Emb.9', 'Emb.10', 'Emb.11'] ,loc=1)
        plt.xlabel('Time [day]', size=12)
        plt.ylabel('Volume [$\mu m^{3}$]', size=12)
        if Time[-1] != 0:
            plt.xlim([0.0, round(Time[-1], 1)])
        plt.ylim([0.0, np.max(np.array(MeanVol)) + 1000])
        filename = self.opbase + self.psep + 'MeanVolume_all.pdf'
        plt.savefig(filename)

        plt.figure()
        for num in range(len(StdVol)):
            colors = cmap(float(num) / len(StdVol))
            plt.plot(Time, StdVol[num], color=colors, alpha=0.8, linewidth=1.0)
        #plt.legend(['Emb.1', 'Emb.2', 'Emb.3', 'Emb.4', 'Emb.5', 'Emb.6', 'Emb.7', 'Emb.8', 'Emb.9', 'Emb.10', 'Emb.11'] ,loc=1)
        plt.xlabel('Time [day]', size=12)
        plt.ylabel('Volume (standard deviation) [$\mu m^{3}$]', size=12)
        if Time[-1] != 0:
            plt.xlim([0.0, round(Time[-1], 1)])
        plt.ylim([0.0, np.max(np.array(StdVol)) + 1000])
        filename = self.opbase + self.psep + 'StdVolume_all.pdf'
        plt.savefig(filename)


    def graph_draw_surface(self, Time, SumArea, MeanArea, StdArea):
        SumArea = np.array(SumArea) * self.scale
        MeanArea = np.array(MeanArea) * self.scale
        StdArea = np.array(StdArea) * self.scale

        # Surface Mean & SD
        cmap =  plt.get_cmap('Paired')
        plt.figure()
        for num in range(len(SumArea)):
            colors = cmap(float(num) / len(SumArea))
            plt.plot(Time, MeanArea[num], color=colors, alpha=0.8, linewidth=1.0)
        #plt.legend(['Emb.1', 'Emb.2', 'Emb.3', 'Emb.4', 'Emb.5', 'Emb.6', 'Emb.7', 'Emb.8', 'Emb.9', 'Emb.10', 'Emb.11'] ,loc=1)
        plt.xlabel('Time [day]', size=12)
        plt.ylabel('Surface Area [$\mu m^{2}$]', size=12)
        if Time[-1] != 0:
            plt.xlim([0.0, round(Time[-1], 1)])
        plt.ylim([0.0, np.max(np.array(MeanArea)) + 1000])
        filename = self.opbase + self.psep + 'MeanSurface_all.pdf'
        plt.savefig(filename)

        cmap =  plt.get_cmap('Paired')
        plt.figure()
        for num in range(len(SumArea)):
            colors = cmap(float(num) / len(SumArea))
            plt.plot(Time, StdArea[num], color=colors, alpha=0.8, linewidth=1.0)
        #plt.legend(['Emb.1', 'Emb.2', 'Emb.3', 'Emb.4', 'Emb.5', 'Emb.6', 'Emb.7', 'Emb.8', 'Emb.9', 'Emb.10', 'Emb.11'] ,loc=1)
        plt.xlabel('Time [day]', size=12)
        plt.ylabel('Surface Area (standard deviation) [$\mu m^{2}$]', size=12)
        if Time[-1] != 0:
            plt.xlim([0.0, round(Time[-1], 1)])
        plt.ylim([0.0, np.max(np.array(StdArea)) + 1000])
        filename = self.opbase + self.psep + 'StdSurface_all.pdf'
        plt.savefig(filename)

    def graph_draw_surface_volume(self, Time, SumArea, MeanArea, StdArea, SumVol, MeanVol, StdVol):
        SumArea = np.array(SumArea) * self.scale
        MeanArea = np.array(MeanArea) * self.scale
        StdArea = np.array(StdArea) * self.scale
        SumVol = np.array(SumVol) * self.scale
        MeanVol = np.array(MeanVol) * self.scale
        StdVol = np.array(StdVol) * self.scale

        assert len(SumArea) == len(SumVol)

        # Surface Mean & SD
        cmap =  plt.get_cmap('Paired')
        plt.figure()
        for num in range(len(SumArea)):
            colors = cmap(float(num) / len(SumArea))
            plt.plot(Time, MeanArea[num] / MeanVol[num], color=colors, alpha=0.8, linewidth=1.0)
        #plt.legend(['Emb.1', 'Emb.2', 'Emb.3', 'Emb.4', 'Emb.5', 'Emb.6', 'Emb.7', 'Emb.8', 'Emb.9', 'Emb.10', 'Emb.11'] ,loc=4)
        plt.xlabel('Time [day]', size=12)
        plt.ylabel('Specific Surface Area [$\mu m^{-1}$]', size=12)
        if Time[-1] != 0:
            plt.xlim([0.0, round(Time[-1], 1)])
        plt.ylim([0.0, np.max(MeanArea / MeanVol) + 0.05])
        filename = self.opbase + self.psep + 'MeanSurfaceVolume_all.pdf'
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


    def graph_draw_lfunction(self, cent_x, cent_y, cent_z):
        roi = {}
        roi_pixel_num = {}
        center = (self.x/2, self.y/2, self.z/2)
        # radius_list = [i for i in range(int(self.z/2))]
        # for r in radius_list:
        #     roi[r] = []
        #     roi_pixel_num[r] = 0
        # for x in range(self.x):
        #     for y in range(self.y):
        #         for z in range(self.z):
        #             for r in radius_list:
        #                 if (x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2 < r ** 2 and \
        #                     (x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2 >= (r - 1) ** 2:
        #                     roi[r].append([x, y, z])
        #                     roi_pixel_num[r] += 1
        cmap =  plt.get_cmap('Paired')
        plt.figure(figsize=(10, 8))
        radius_len = 25
        radius_list = [i for i in range(radius_len)]
        with open(os.path.join(self.opbase, 'apf.csv'), 'w') as f:
            writer = csv.writer(f, lineterminator='\n')
            for num in range(len(cent_x)):
                for r in radius_list:
                    roi[r] = []
                    roi_pixel_num[r] = 0
                center = (np.mean([np.mean(i) for i in cent_x[num]]), np.mean([np.mean(i) for i in cent_y[num]]), np.mean([np.mean(i) for i in cent_z[num]]))
                #print(center)
                for x in range(int(center[0] + radius_len) + 1):
                    for y in range(int(center[1] + radius_len) + 1):
                        for z in range(int(center[2] + radius_len) + 1):
                            for r in radius_list:
                                if (x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2 < r ** 2 and \
                                    (x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2 >= (r - 1) ** 2:
                                    roi[r].append([x, y, z])
                                    roi_pixel_num[r] += 1
                self.roi_pixel_num = 0
                self.density = 0
                colors = cmap(float(num) / len(cent_x))
                apf = [self.volume_density(roi, roi_pixel_num, r, cent_x[num], cent_y[num], cent_z[num]) for r in radius_list]
                plt.plot(radius_list, apf, color=colors, alpha=0.8, linewidth=1.0)
                writer.writerow(apf)
                #print('Embryo num. {} complete.'.format(num + 1))
        plt.xlabel('Radius [pixel]', size=12)
        plt.ylabel('Space fill factor', size=12)
        plt.xlim([0, 25])
        #plt.legend(['Emb.1', 'Emb.2', 'Emb.3', 'Emb.4', 'Emb.5', 'Emb.6', 'Emb.7', 'Emb.8', 'Emb.9', 'Emb.10', 'Emb.11'] ,loc=1)
        filename = self.opbase + self.psep + 'L-function.pdf'
        plt.savefig(filename)

    def volume_density(self, roi, roi_pixel_num, radius, cent_x, cent_y, cent_z):
        density = 0
        for t in zip(cent_x, cent_y, cent_z):
            for cent in zip(t[0], t[1], t[2]):
                if [int(cent[0]), int(cent[1]), int(cent[2])] in roi[radius]:
                    density += 1
        self.density += density
        self.roi_pixel_num += roi_pixel_num[radius]
        try:
            return float(self.density) / float(self.roi_pixel_num)
        except:
            return 0.0


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
    ap = ArgumentParser(description='python graph_draw_all.py')
    ap.add_argument('--indir', '-i', nargs='?', default='criteria.csv', help='Specify input files (format : csv)')
    ap.add_argument('--outdir', '-o', nargs='?', default='extract_figs_all', help='Specify output files directory for create figures')
    ap.add_argument('--roi', '-r', type=int, default=0, help='Specify ROI GT')
    args = ap.parse_args()
    argvs = sys.argv
    psep = '/'
    opbase = createOpbase(args.outdir)

    file_name = glob(os.path.join(args.indir, '*', '*', '*', 'extract_criteria_*', 'criteria.csv'))
    print('the number of file: {}'.format(len(file_name)))

    all_count = []
    all_vol_sum, all_vol_mean, all_vol_std = [], [], []
    all_sur_sum, all_sur_mean, all_sur_std = [], [], []
    all_cent_x, all_cent_y, all_cent_z = [], [], []

    for fn in file_name:
        # each criterion
        SumVol, SumArea, Count = [], [], []
        MeanVol, MeanArea, StdVol, StdArea = [], [], [], []
        Cent_X, Cent_Y, Cent_Z = [], [], []

        # import csv
        f = open(fn, 'r')
        data = csv.reader(f)
        l = []
        for i in data:
            l.append(i)
        f.close()
        l.pop(0)

        for c in range(len(l)):
            if l[c][1] is '':
                Count.append(np.nan)
                SumVol.append(np.nan)
                MeanVol.append(np.nan)
                StdVol.append(np.nan)
                SumArea.append(np.nan)
                MeanArea.append(np.nan)
                StdArea.append(np.nan)
                Cent_X.append([np.nan])
                Cent_Y.append([np.nan])
                Cent_Z.append([np.nan])

            elif int(l[c][1]) > 0:
                Count.append(int(l[c][1]))
                SumVol.append(float(l[c][2]))
                MeanVol.append(float(l[c][3]))
                StdVol.append(float(l[c][4]))
                SumArea.append(float(l[c][5]))
                MeanArea.append(float(l[c][6]))
                StdArea.append(float(l[c][7]))
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
                StdVol.append(0)
                SumArea.append(0)
                MeanArea.append(0)
                StdArea.append(0)
                Cent_X.append([0])
                Cent_Y.append([0])
                Cent_Z.append([0])

        all_vol_sum.append(SumVol)
        all_sur_sum.append(SumArea)
        all_vol_mean.append(MeanVol)
        all_sur_mean.append(MeanArea)
        all_vol_std.append(StdVol)
        all_sur_std.append(StdArea)
        all_count.append(Count)
        all_cent_x.append(Cent_X)
        all_cent_y.append(Cent_Y)
        all_cent_z.append(Cent_Z)


    without = True
    if without:
        thr = 100 #4
        pop_list = []
        for i in range(len(all_count)):
            if np.unique((np.array(all_count[i][:100]) > thr) * 1)[0] == 0 and \
               len(np.unique((np.array(all_count[i][:100]) > thr) * 1)) < 2:
                pass
            else:
                pop_list.append(i)
                with open('without_embryo_filename.txt', 'a') as f:
                    print(file_name[i])
                    f.write('{}\n'.format(file_name[i][28:]))

        print('number of faiulure segmentation: {}'.format(len(pop_list)))
        c = 0
        for p in pop_list:
            all_vol_sum.pop(p - c)
            all_sur_sum.pop(p - c)
            all_vol_mean.pop(p - c)
            all_sur_mean.pop(p - c)
            all_vol_std.pop(p - c)
            all_sur_std.pop(p - c)
            all_count.pop(p - c)
            c += 1

        

    # Time Scale
    dt = 10 / float(60 * 24)
    count_max = 0
    for i in range(len(all_count)):
        count_max = np.max([len(all_count[i]), count_max])
    #Time = [dt*x for x in range(len(Count))]
    Time = [dt*x for x in range(count_max)]

    gd = GraphDrawAll(opbase, args.roi, file_name)
    gd.graph_draw_number(Time, all_count)
    #gd.graph_draw_synchronicity(Time, all_count)
    gd.graph_draw_volume(Time, all_vol_sum, all_vol_mean, all_vol_std)
    gd.graph_draw_surface(Time, all_sur_sum, all_sur_mean, all_sur_std)
    #gd.graph_draw_surface_volume(Time, all_sur_sum, all_sur_mean, all_sur_std, all_vol_sum, all_vol_mean, all_vol_std)
    #gd.graph_draw_lfunction(all_cent_x, all_cent_y, all_cent_z)
    #plt.show()
