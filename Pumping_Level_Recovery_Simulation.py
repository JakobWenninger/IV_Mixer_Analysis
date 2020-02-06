from IV_Class import *
from scipy.special import jv

filename = r"C:\Users\Jakob\Synchronised_Data\Master_Manchester\SIS_Junction_DC_IV_Curve_Measurment\AlOx_2019_02\Data\Holder5Device19_13_magnet.csv"

IV = IV_Response(filename,areaDoubleJunction)

#plot IV curve added with itself at a voltage offset
current = IV.offsetCorrectedBinedIVData[1][200:] + IV.offsetCorrectedBinedIVData[1][:1801]
plt.plot(current) # without voltage axis
plt.show()
plt.plot(IV.offsetCorrectedBinedIVData[0][200:],current) # with voltage axis
plt.show()#TODO not clear why this voltage is centred? -> Artefact due to not including J(alpha)?

#include J(alpha)
alpha = .5 #varying alpha, 1 1./3 .5 , 2
firstTerm =jv(0,alpha)*jv(0,alpha)*IV.offsetCorrectedBinedIVData[1][:1801] # 
secondTerm =jv(1,alpha)*jv(1,alpha)*IV.offsetCorrectedBinedIVData[1][200:]
current =  firstTerm + secondTerm 
plt.plot(IV.offsetCorrectedBinedIVData[0][:1801],current,label='current') # without voltage axis
plt.plot(IV.offsetCorrectedBinedIVData[0][:1801],firstTerm,label='1st') 
plt.plot(IV.offsetCorrectedBinedIVData[0][:1801],secondTerm,label='2nd') 
plt.vlines(IV.gapVoltage, -50,50)
plt.vlines(-IV.gapVoltage, -50,50)
plt.legend(loc='best', shadow=False,ncol=1)
plt.show()

