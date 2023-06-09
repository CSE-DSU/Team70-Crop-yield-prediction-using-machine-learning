
# importing necessary libraries 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pickle

final = ['Apple', 'Banana', 'Blackgram', 'Coconut', 'Coffee', 'Cotton(lint)',
       'Grapes', 'Jute', 'Lentil', 'Maize', 'Mango', 'Moth', 'Orange',
       'Papaya', 'Pome Granet', 'Rice', 'Water Melon']



filename = 'random_forest_crop_rec.pkl'
with open(filename, 'rb') as file:
    multi_target_forest = pickle.load(file)

filename = 'random_forest_model.pkl'
with open(filename, 'rb') as file:
    model = pickle.load(file)

# model.score(x_test,y_test)

state_districts = {
    'Andaman and Nicobar Islands': ['NICOBARS' ,'NORTH AND MIDDLE ANDAMAN', 'SOUTH ANDAMANS'],
    'Andhra Pradesh': ['ANANTAPUR','CHITTOOR', 'EAST GODAVARI', 'GUNTUR', 'KADAPA', 'KRISHNA', 'KURNOOL', 'PRAKASAM', 'SPSR NELLORE', 'SRIKAKULAM', 'VISAKHAPATANAM', 'VIZIANAGARAM', 'WEST GODAVARI'],
    'Arunachal Pradesh': ['ANJAW', 'CHANGLANG', 'DIBANG VALLEY', 'EAST KAMENG', 'EAST SIANG',
    'KURUNG KUMEY', 'LOHIT', 'LONGDING', 'LOWER DIBANG VALLEY', 'LOWER SUBANSIRI',
    'NAMSAI', 'PAPUM PARE', 'TAWANG', 'TIRAP', 'UPPER SIANG', 'UPPER SUBANSIRI',
    'WEST KAMENG', 'WEST SIANG'],
    'Assam': ['BAKSA', 'BARPETA', 'BONGAIGAON', 'CACHAR', 'CHIRANG', 'DARRANG', 'DHEMAJI',
 'DHUBRI', 'DIBRUGARH', 'DIMA HASAO', 'GOALPARA', 'GOLAGHAT', 'HAILAKANDI',
 'JORHAT', 'KAMRUP', 'KAMRUP METRO', 'KARBI ANGLONG', 'KARIMGANJ', 'KOKRAJHAR',
 'LAKHIMPUR', 'MARIGAON', 'NAGAON', 'NALBARI', 'SIVASAGAR', 'SONITPUR',
 'TINSUKIA', 'UDALGURI'],
    'Bihar': ['ARARIA', 'ARWAL', 'AURANGABAD', 'BANKA', 'BEGUSARAI', 'BHAGALPUR', 'BHOJPUR',
 'BUXAR', 'DARBHANGA', 'GAYA', 'GOPALGANJ', 'JAMUI', 'JEHANABAD',
 'KAIMUR (BHABUA)', 'KATIHAR', 'KHAGARIA', 'KISHANGANJ' ,'LAKHISARAI',
 'MADHEPURA', 'MADHUBANI' ,'MUNGER', 'MUZAFFARPUR' ,'NALANDA' ,'NAWADA',
 'PASHCHIM CHAMPARAN' ,'PATNA' ,'PURBI CHAMPARAN', 'PURNIA', 'ROHTAS',
 'SAHARSA' ,'SAMASTIPUR', 'SARAN' ,'SHEIKHPURA' ,'SHEOHAR', 'SITAMARHI', 'SIWAN',
 'SUPAUL', 'VAISHALI'],
    'Chandigarh': ['CHANDIGARH'],
    'Chhattisgarh': ['BALOD', 'BALODA BAZAR', 'BALRAMPUR', 'BASTAR', 'BEMETARA', 'BIJAPUR',
 'BILASPUR', 'DANTEWADA', 'DHAMTARI', 'DURG', 'GARIYABAND', 'JANJGIR-CHAMPA',
 'JASHPUR', 'KABIRDHAM', 'KANKER', 'KONDAGAON', 'KORBA', 'KOREA', 'MAHASAMUND',
 'MUNGELI', 'NARAYANPUR', 'RAIGARH', 'RAIPUR', 'RAJNANDGAON', 'SUKMA',
 'SURAJPUR', 'SURGUJA'],
    'Dadra and Nagar Haveli': ['DADRA AND NAGAR HAVELI'],
    'Goa': ['NORTH GOA', 'SOUTH GOA'],
    'Gujarat': ['AHMADABAD', 'AMRELI', 'ANAND', 'BANAS KANTHA', 'BHARUCH', 'BHAVNAGAR', 'DANG',
 'DOHAD', 'GANDHINAGAR', 'JAMNAGAR', 'JUNAGADH', 'KACHCHH', 'KHEDA', 'MAHESANA',
 'NARMADA', 'NAVSARI', 'PANCH MAHALS', 'PATAN', 'PORBANDAR', 'RAJKOT',
 'SABAR KANTHA', 'SURAT', 'SURENDRANAGAR', 'TAPI', 'VADODARA', 'VALSAD'],
    'Haryana': ['AMBALA', 'BHIWANI', 'FARIDABAD' ,'FATEHABAD', 'GURGAON', 'HISAR', 'JHAJJAR',
 'JIND', 'KAITHAL', 'KARNAL', 'KURUKSHETRA', 'MAHENDRAGARH', 'MEWAT', 'PALWAL',
 'PANCHKULA', 'PANIPAT', 'REWARI', 'ROHTAK', 'SIRSA', 'SONIPAT', 'YAMUNANAGAR'],
    'Himachal Pradesh': ['BILASPUR', 'CHAMBA', 'HAMIRPUR', 'KANGRA', 'KINNAUR', 'KULLU',
 'LAHUL AND SPITI', 'MANDI', 'SHIMLA', 'SIRMAUR', 'SOLAN', 'UNA'],
    'Jammu and Kashmir': ['ANANTNAG', 'BADGAM', 'BANDIPORA', 'BARAMULLA', 'DODA', 'GANDERBAL', 'JAMMU',
 'KARGIL', 'KATHUA', 'KISHTWAR', 'KULGAM', 'KUPWARA', 'LEH LADAKH', 'POONCH',
 'PULWAMA', 'RAJAURI', 'RAMBAN', 'REASI', 'SAMBA', 'SHOPIAN', 'SRINAGAR',
 'UDHAMPUR'],
    'Jharkhand': ['BOKARO', 'CHATRA', 'DEOGHAR', 'DHANBAD', 'DUMKA', 'EAST SINGHBUM', 'GARHWA',
 'GIRIDIH', 'GODDA', 'GUMLA', 'HAZARIBAGH', 'JAMTARA', 'KHUNTI', 'KODERMA',
 'LATEHAR', 'LOHARDAGA', 'PAKUR', 'PALAMU', 'RAMGARH', 'RANCHI', 'SAHEBGANJ',
 'SARAIKELA KHARSAWAN', 'SIMDEGA', 'WEST SINGHBHUM'],
    'Karnataka': ['BAGALKOT', 'BANGALORE RURAL', 'BELGAUM', 'BELLARY', 'BENGALURU URBAN',
 'BIDAR', 'BIJAPUR', 'CHAMARAJANAGAR', 'CHIKBALLAPUR', 'CHIKMAGALUR',
 'CHITRADURGA', 'DAKSHIN KANNAD', 'DAVANGERE'],
}

data = {'Area': [0], 'District_Name_24 PARAGANAS NORTH': [0], 'District_Name_24 PARAGANAS SOUTH': [0], 'District_Name_ADILABAD': [0], 'District_Name_AGAR MALWA': [0], 'District_Name_AGRA': [0], 'District_Name_AHMADABAD': [0], 'District_Name_AHMEDNAGAR': [0], 'District_Name_AIZAWL': [0], 'District_Name_AJMER': [0], 'District_Name_AKOLA': [0], 'District_Name_ALAPPUZHA': [0], 'District_Name_ALIGARH': [0], 'District_Name_ALIRAJPUR': [0], 'District_Name_ALLAHABAD': [0], 'District_Name_ALMORA': [0], 'District_Name_ALWAR': [0], 'District_Name_AMBALA': [0], 'District_Name_AMBEDKAR NAGAR': [0], 'District_Name_AMETHI': [0], 'District_Name_AMRAVATI': [0], 'District_Name_AMRELI': [0], 'District_Name_AMRITSAR': [0], 'District_Name_AMROHA': [0], 'District_Name_ANAND': [0], 'District_Name_ANANTAPUR': [0], 'District_Name_ANANTNAG': [0], 'District_Name_ANJAW': [0], 'District_Name_ANUGUL': [0], 'District_Name_ANUPPUR': [0], 'District_Name_ARARIA': [0], 'District_Name_ARIYALUR': [0], 'District_Name_ARWAL': [0], 'District_Name_ASHOKNAGAR': [0], 'District_Name_AURAIYA': [0], 'District_Name_AURANGABAD': [0], 'District_Name_AZAMGARH': [0], 'District_Name_BADGAM': [0], 'District_Name_BAGALKOT': [0], 'District_Name_BAGESHWAR': [0], 'District_Name_BAGHPAT': [0], 'District_Name_BAHRAICH': [0], 'District_Name_BAKSA': [0], 'District_Name_BALAGHAT': [0], 'District_Name_BALANGIR': [0], 'District_Name_BALESHWAR': [0], 'District_Name_BALLIA': [0], 'District_Name_BALOD': [0], 'District_Name_BALODA BAZAR': [0], 'District_Name_BALRAMPUR': [0], 'District_Name_BANAS KANTHA': [0], 'District_Name_BANDA': [0], 'District_Name_BANDIPORA': [0], 'District_Name_BANGALORE RURAL': [0], 'District_Name_BANKA': [0], 'District_Name_BANKURA': [0], 'District_Name_BANSWARA': [0], 'District_Name_BARABANKI': 
[0], 'District_Name_BARAMULLA': [0], 'District_Name_BARAN': [0], 'District_Name_BARDHAMAN': [0], 'District_Name_BAREILLY': [0], 'District_Name_BARGARH': [0], 'District_Name_BARMER': [0], 'District_Name_BARNALA': [0], 'District_Name_BARPETA': [0], 'District_Name_BARWANI': [0], 'District_Name_BASTAR': [0], 'District_Name_BASTI': [0], 'District_Name_BATHINDA': [0], 'District_Name_BEED': [0], 'District_Name_BEGUSARAI': [0], 'District_Name_BELGAUM': [0], 'District_Name_BELLARY': [0], 'District_Name_BEMETARA': [0], 'District_Name_BENGALURU URBAN': [0], 'District_Name_BETUL': [0], 'District_Name_BHADRAK': [0], 'District_Name_BHAGALPUR': [0], 'District_Name_BHANDARA': [0], 'District_Name_BHARATPUR': [0], 'District_Name_BHARUCH': [0], 'District_Name_BHAVNAGAR': [0], 'District_Name_BHILWARA': [0], 'District_Name_BHIND': [0], 'District_Name_BHIWANI': [0], 'District_Name_BHOJPUR': [0], 'District_Name_BHOPAL': [0], 'District_Name_BIDAR': [0], 'District_Name_BIJAPUR': [0], 'District_Name_BIJNOR': [0], 'District_Name_BIKANER': [0], 'District_Name_BILASPUR': [0], 'District_Name_BIRBHUM': [0], 'District_Name_BISHNUPUR': [0], 'District_Name_BOKARO': [0], 'District_Name_BONGAIGAON': [0], 'District_Name_BOUDH': [0], 'District_Name_BUDAUN': [0], 'District_Name_BULANDSHAHR': [0], 'District_Name_BULDHANA': [0], 'District_Name_BUNDI': [0], 'District_Name_BURHANPUR': [0], 'District_Name_BUXAR': [0], 'District_Name_CACHAR': [0], 'District_Name_CHAMARAJANAGAR': [0], 'District_Name_CHAMBA': [0], 'District_Name_CHAMOLI': [0], 'District_Name_CHAMPAWAT': [0], 'District_Name_CHAMPHAI': [0], 'District_Name_CHANDAULI': [0], 'District_Name_CHANDEL': [0], 'District_Name_CHANDIGARH': [0], 'District_Name_CHANDRAPUR': [0], 'District_Name_CHANGLANG': [0], 'District_Name_CHATRA': [0], 'District_Name_CHHATARPUR': [0], 'District_Name_CHHINDWARA': [0], 'District_Name_CHIKBALLAPUR': [0], 'District_Name_CHIKMAGALUR': [0], 'District_Name_CHIRANG': [0], 'District_Name_CHITRADURGA': [0], 'District_Name_CHITRAKOOT': [0], 'District_Name_CHITTOOR': [0], 'District_Name_CHITTORGARH': [0], 
'District_Name_CHURACHANDPUR': [0], 'District_Name_CHURU': [0], 'District_Name_COIMBATORE': [0], 'District_Name_COOCHBEHAR': [0], 'District_Name_CUDDALORE': [0], 'District_Name_CUTTACK': [0], 'District_Name_DADRA AND NAGAR HAVELI': [0], 'District_Name_DAKSHIN KANNAD': [0], 'District_Name_DAMOH': [0], 'District_Name_DANG': [0], 'District_Name_DANTEWADA': [0], 'District_Name_DARBHANGA': [0], 'District_Name_DARJEELING': [0], 'District_Name_DARRANG': [0], 'District_Name_DATIA': [0], 'District_Name_DAUSA': [0], 'District_Name_DAVANGERE': [0], 'District_Name_DEHRADUN': [0], 'District_Name_DEOGARH': [0], 'District_Name_DEOGHAR': [0], 'District_Name_DEORIA': [0], 'District_Name_DEWAS': [0], 'District_Name_DHALAI': [0], 'District_Name_DHAMTARI': [0], 'District_Name_DHANBAD': [0], 'District_Name_DHAR': [0], 'District_Name_DHARMAPURI': [0], 'District_Name_DHARWAD': [0], 'District_Name_DHEMAJI': [0], 'District_Name_DHENKANAL': [0], 'District_Name_DHOLPUR': [0], 'District_Name_DHUBRI': [0], 'District_Name_DHULE': [0], 'District_Name_DIBANG VALLEY': [0], 'District_Name_DIBRUGARH': [0], 'District_Name_DIMA HASAO': [0], 'District_Name_DIMAPUR': [0], 'District_Name_DINAJPUR DAKSHIN': [0], 'District_Name_DINAJPUR UTTAR': [0], 'District_Name_DINDIGUL': [0], 'District_Name_DINDORI': [0], 'District_Name_DODA': [0], 'District_Name_DOHAD': [0], 'District_Name_DUMKA': [0], 'District_Name_DUNGARPUR': [0], 'District_Name_DURG': [0], 'District_Name_EAST DISTRICT': [0], 'District_Name_EAST GARO HILLS': [0], 'District_Name_EAST GODAVARI': [0], 'District_Name_EAST JAINTIA HILLS': [0], 'District_Name_EAST KAMENG': [0], 'District_Name_EAST KHASI HILLS': [0], 'District_Name_EAST SIANG': [0], 'District_Name_EAST SINGHBUM': [0], 'District_Name_ERNAKULAM': [0], 'District_Name_ERODE': [0], 'District_Name_ETAH': [0], 'District_Name_ETAWAH': [0], 'District_Name_FAIZABAD': [0], 'District_Name_FARIDABAD': [0], 'District_Name_FARIDKOT': [0], 'District_Name_FARRUKHABAD': [0], 'District_Name_FATEHABAD': [0], 'District_Name_FATEHGARH SAHIB': [0], 'District_Name_FATEHPUR': [0], 'District_Name_FAZILKA': [0], 'District_Name_FIROZABAD': [0], 'District_Name_FIROZEPUR': [0], 'District_Name_GADAG': [0], 'District_Name_GADCHIROLI': [0], 'District_Name_GAJAPATI': [0], 'District_Name_GANDERBAL': [0], 'District_Name_GANDHINAGAR': [0], 'District_Name_GANGANAGAR': [0], 'District_Name_GANJAM': [0], 'District_Name_GARHWA': [0], 'District_Name_GARIYABAND': [0], 'District_Name_GAUTAM BUDDHA NAGAR': [0], 'District_Name_GAYA': [0], 'District_Name_GHAZIABAD': [0], 'District_Name_GHAZIPUR': [0], 'District_Name_GIRIDIH': [0], 'District_Name_GOALPARA': [0], 'District_Name_GODDA': [0], 'District_Name_GOLAGHAT': [0], 'District_Name_GOMATI': [0], 'District_Name_GONDA': [0], 'District_Name_GONDIA': [0], 'District_Name_GOPALGANJ': [0], 'District_Name_GORAKHPUR': [0], 'District_Name_GULBARGA': [0], 'District_Name_GUMLA': [0], 'District_Name_GUNA': [0], 'District_Name_GUNTUR': [0], 'District_Name_GURDASPUR': [0], 'District_Name_GURGAON': [0], 'District_Name_GWALIOR': [0], 'District_Name_HAILAKANDI': [0], 'District_Name_HAMIRPUR': [0], 'District_Name_HANUMANGARH': [0], 'District_Name_HAPUR': [0], 'District_Name_HARDA': [0], 'District_Name_HARDOI': [0], 'District_Name_HARIDWAR': [0], 'District_Name_HASSAN': [0], 'District_Name_HATHRAS': [0], 'District_Name_HAVERI': [0], 'District_Name_HAZARIBAGH': [0], 'District_Name_HINGOLI': [0], 'District_Name_HISAR': [0], 'District_Name_HOOGHLY': [0], 'District_Name_HOSHANGABAD': [0], 'District_Name_HOSHIARPUR': [0], 'District_Name_HOWRAH': [0], 'District_Name_HYDERABAD': [0], 'District_Name_IDUKKI': [0], 'District_Name_IMPHAL EAST': [0], 'District_Name_IMPHAL WEST': [0], 'District_Name_INDORE': [0], 'District_Name_JABALPUR': [0], 'District_Name_JAGATSINGHAPUR': [0], 'District_Name_JAIPUR': [0], 'District_Name_JAISALMER': [0], 'District_Name_JAJAPUR': [0], 'District_Name_JALANDHAR': [0], 'District_Name_JALAUN': [0], 'District_Name_JALGAON': [0], 'District_Name_JALNA': [0], 'District_Name_JALORE': [0], 'District_Name_JALPAIGURI': [0], 'District_Name_JAMMU': [0], 'District_Name_JAMNAGAR': [0], 'District_Name_JAMTARA': [0], 'District_Name_JAMUI': [0], 'District_Name_JANJGIR-CHAMPA': [0], 'District_Name_JASHPUR': [0], 'District_Name_JAUNPUR': [0], 'District_Name_JEHANABAD': [0], 'District_Name_JHABUA': [0], 'District_Name_JHAJJAR': [0], 'District_Name_JHALAWAR': [0], 'District_Name_JHANSI': [0], 'District_Name_JHARSUGUDA': [0], 'District_Name_JHUNJHUNU': [0], 'District_Name_JIND': [0], 'District_Name_JODHPUR': [0], 'District_Name_JORHAT': [0], 'District_Name_JUNAGADH': [0], 'District_Name_KABIRDHAM': [0], 'District_Name_KACHCHH': [0], 'District_Name_KADAPA': [0], 'District_Name_KAIMUR (BHABUA)': [0], 'District_Name_KAITHAL': [0], 'District_Name_KALAHANDI': [0], 'District_Name_KAMRUP': [0], 'District_Name_KAMRUP METRO': [0], 'District_Name_KANCHIPURAM': [0], 'District_Name_KANDHAMAL': [0], 'District_Name_KANGRA': [0], 'District_Name_KANKER': [0], 'District_Name_KANNAUJ': [0], 'District_Name_KANNIYAKUMARI': [0], 'District_Name_KANNUR': [0], 'District_Name_KANPUR DEHAT': [0], 'District_Name_KANPUR NAGAR': [0], 'District_Name_KAPURTHALA': [0], 'District_Name_KARAIKAL': [0], 'District_Name_KARAULI': [0], 'District_Name_KARBI ANGLONG': [0], 'District_Name_KARGIL': [0], 'District_Name_KARIMGANJ': [0], 'District_Name_KARIMNAGAR': [0], 'District_Name_KARNAL': [0], 'District_Name_KARUR': [0], 'District_Name_KASARAGOD': [0], 'District_Name_KASGANJ': [0], 'District_Name_KATHUA': [0], 'District_Name_KATIHAR': [0], 'District_Name_KATNI': [0], 'District_Name_KAUSHAMBI': [0], 
'District_Name_KENDRAPARA': [0], 'District_Name_KENDUJHAR': [0], 'District_Name_KHAGARIA': [0], 'District_Name_KHAMMAM': [0], 'District_Name_KHANDWA': [0], 'District_Name_KHARGONE': [0], 'District_Name_KHEDA': [0], 'District_Name_KHERI': [0], 'District_Name_KHORDHA': [0], 'District_Name_KHOWAI': [0], 'District_Name_KHUNTI': [0], 'District_Name_KINNAUR': [0], 'District_Name_KIPHIRE': [0], 'District_Name_KISHANGANJ': [0], 'District_Name_KISHTWAR': [0], 'District_Name_KODAGU': [0], 'District_Name_KODERMA': [0], 'District_Name_KOHIMA': [0], 'District_Name_KOKRAJHAR': [0], 'District_Name_KOLAR': [0], 'District_Name_KOLASIB': [0], 'District_Name_KOLHAPUR': [0], 'District_Name_KOLLAM': [0], 'District_Name_KONDAGAON': [0], 'District_Name_KOPPAL': [0], 'District_Name_KORAPUT': [0], 'District_Name_KORBA': [0], 'District_Name_KOREA': [0], 'District_Name_KOTA': [0], 'District_Name_KOTTAYAM': [0], 'District_Name_KOZHIKODE': [0], 'District_Name_KRISHNA': [0], 'District_Name_KRISHNAGIRI': [0], 'District_Name_KULGAM': [0], 'District_Name_KULLU': [0], 'District_Name_KUPWARA': [0], 'District_Name_KURNOOL': [0], 'District_Name_KURUKSHETRA': [0], 'District_Name_KURUNG KUMEY': [0], 'District_Name_KUSHI NAGAR': [0], 'District_Name_LAHUL AND SPITI': [0], 'District_Name_LAKHIMPUR': [0], 'District_Name_LAKHISARAI': [0], 'District_Name_LALITPUR': [0], 'District_Name_LATEHAR': [0], 'District_Name_LATUR': [0], 'District_Name_LAWNGTLAI': [0], 'District_Name_LEH LADAKH': [0], 'District_Name_LOHARDAGA': [0], 'District_Name_LOHIT': [0], 'District_Name_LONGDING': [0], 'District_Name_LONGLENG': [0], 'District_Name_LOWER DIBANG VALLEY': [0], 'District_Name_LOWER SUBANSIRI': [0], 'District_Name_LUCKNOW': [0], 'District_Name_LUDHIANA': [0], 'District_Name_LUNGLEI': [0], 'District_Name_MADHEPURA': [0], 'District_Name_MADHUBANI': [0], 'District_Name_MADURAI': [0], 'District_Name_MAHARAJGANJ': [0], 'District_Name_MAHASAMUND': [0], 'District_Name_MAHBUBNAGAR': [0], 'District_Name_MAHE': [0], 'District_Name_MAHENDRAGARH': [0], 'District_Name_MAHESANA': [0], 'District_Name_MAHOBA': [0], 'District_Name_MAINPURI': [0], 'District_Name_MALAPPURAM': [0], 'District_Name_MALDAH': [0], 'District_Name_MALKANGIRI': [0], 'District_Name_MAMIT': [0], 'District_Name_MANDI': [0], 'District_Name_MANDLA': [0], 'District_Name_MANDSAUR': [0], 'District_Name_MANDYA': [0], 'District_Name_MANSA': [0], 'District_Name_MARIGAON': [0], 'District_Name_MATHURA': [0], 'District_Name_MAU': [0], 'District_Name_MAYURBHANJ': [0], 'District_Name_MEDAK': [0], 'District_Name_MEDINIPUR EAST': [0], 'District_Name_MEDINIPUR WEST': [0], 'District_Name_MEERUT': [0], 'District_Name_MEWAT': [0], 'District_Name_MIRZAPUR': [0], 'District_Name_MOGA': [0], 'District_Name_MOKOKCHUNG': [0], 'District_Name_MON': [0], 'District_Name_MORADABAD': [0], 'District_Name_MORENA': [0], 'District_Name_MUKTSAR': [0], 'District_Name_MUMBAI': [0], 'District_Name_MUNGELI': [0], 'District_Name_MUNGER': [0], 'District_Name_MURSHIDABAD': [0], 'District_Name_MUZAFFARNAGAR': [0], 'District_Name_MUZAFFARPUR': [0], 'District_Name_MYSORE': [0], 'District_Name_NABARANGPUR': [0], 'District_Name_NADIA': [0], 'District_Name_NAGAON': [0], 'District_Name_NAGAPATTINAM': [0], 'District_Name_NAGAUR': [0], 'District_Name_NAGPUR': [0], 'District_Name_NAINITAL': [0], 'District_Name_NALANDA': [0], 'District_Name_NALBARI': [0], 'District_Name_NALGONDA': [0], 'District_Name_NAMAKKAL': [0], 'District_Name_NAMSAI': [0], 'District_Name_NANDED': [0], 'District_Name_NANDURBAR': [0], 'District_Name_NARAYANPUR': [0], 'District_Name_NARMADA': [0], 'District_Name_NARSINGHPUR': [0], 'District_Name_NASHIK': [0], 'District_Name_NAVSARI': [0], 'District_Name_NAWADA': [0], 
'District_Name_NAWANSHAHR': [0], 'District_Name_NAYAGARH': [0], 'District_Name_NEEMUCH': [0], 'District_Name_NICOBARS': [0], 'District_Name_NIZAMABAD': [0], 'District_Name_NORTH AND MIDDLE ANDAMAN': [0], 'District_Name_NORTH DISTRICT': [0], 'District_Name_NORTH GARO HILLS': [0], 'District_Name_NORTH GOA': [0], 'District_Name_NORTH TRIPURA': [0], 'District_Name_NUAPADA': [0], 'District_Name_OSMANABAD': [0], 'District_Name_PAKUR': [0], 'District_Name_PALAKKAD': [0], 'District_Name_PALAMU': [0], 'District_Name_PALGHAR': [0], 'District_Name_PALI': [0], 'District_Name_PALWAL': [0], 'District_Name_PANCH MAHALS': [0], 'District_Name_PANCHKULA': [0], 'District_Name_PANIPAT': [0], 'District_Name_PANNA': [0], 'District_Name_PAPUM PARE': [0], 'District_Name_PARBHANI': [0], 'District_Name_PASHCHIM CHAMPARAN': [0], 'District_Name_PATAN': [0], 'District_Name_PATHANAMTHITTA': [0], 'District_Name_PATHANKOT': [0], 'District_Name_PATIALA': [0], 'District_Name_PATNA': [0], 'District_Name_PAURI GARHWAL': [0], 'District_Name_PERAMBALUR': [0], 'District_Name_PEREN': [0], 'District_Name_PHEK': [0], 'District_Name_PILIBHIT': [0], 'District_Name_PITHORAGARH': [0], 'District_Name_PONDICHERRY': [0], 'District_Name_POONCH': [0], 'District_Name_PORBANDAR': [0], 'District_Name_PRAKASAM': [0], 'District_Name_PRATAPGARH': [0], 'District_Name_PUDUKKOTTAI': [0], 'District_Name_PULWAMA': [0], 'District_Name_PUNE': [0], 'District_Name_PURBI CHAMPARAN': [0], 'District_Name_PURI': [0], 'District_Name_PURNIA': [0], 'District_Name_PURULIA': [0], 'District_Name_RAE BARELI': [0], 'District_Name_RAICHUR': [0], 'District_Name_RAIGAD': [0], 'District_Name_RAIGARH': [0], 'District_Name_RAIPUR': [0], 'District_Name_RAISEN': [0], 'District_Name_RAJAURI': [0], 'District_Name_RAJGARH': [0], 'District_Name_RAJKOT': [0], 'District_Name_RAJNANDGAON': [0], 'District_Name_RAJSAMAND': [0], 'District_Name_RAMANAGARA': [0], 'District_Name_RAMANATHAPURAM': [0], 'District_Name_RAMBAN': [0], 'District_Name_RAMGARH': [0], 'District_Name_RAMPUR': [0], 'District_Name_RANCHI': [0], 'District_Name_RANGAREDDI': [0], 'District_Name_RATLAM': [0], 'District_Name_RATNAGIRI': [0], 'District_Name_RAYAGADA': [0], 'District_Name_REASI': [0], 'District_Name_REWA': [0], 'District_Name_REWARI': [0], 'District_Name_RI BHOI': [0], 'District_Name_ROHTAK': [0], 'District_Name_ROHTAS': [0], 'District_Name_RUDRA PRAYAG': [0], 'District_Name_RUPNAGAR': [0], 'District_Name_S.A.S NAGAR': [0], 'District_Name_SABAR KANTHA': [0], 'District_Name_SAGAR': [0], 'District_Name_SAHARANPUR': [0], 'District_Name_SAHARSA': [0], 'District_Name_SAHEBGANJ': [0], 'District_Name_SAIHA': [0], 'District_Name_SALEM': [0], 'District_Name_SAMASTIPUR': [0], 'District_Name_SAMBA': [0], 'District_Name_SAMBALPUR': [0], 'District_Name_SAMBHAL': [0], 'District_Name_SANGLI': 
[0], 'District_Name_SANGRUR': [0], 'District_Name_SANT KABEER NAGAR': [0], 'District_Name_SANT RAVIDAS NAGAR': [0], 'District_Name_SARAIKELA KHARSAWAN': [0], 'District_Name_SARAN': [0], 'District_Name_SATARA': [0], 'District_Name_SATNA': [0], 'District_Name_SAWAI MADHOPUR': [0], 'District_Name_SEHORE': [0], 'District_Name_SENAPATI': [0], 'District_Name_SEONI': [0], 'District_Name_SEPAHIJALA': [0], 'District_Name_SERCHHIP': [0], 'District_Name_SHAHDOL': [0], 'District_Name_SHAHJAHANPUR': [0], 'District_Name_SHAJAPUR': [0], 'District_Name_SHAMLI': [0], 'District_Name_SHEIKHPURA': [0], 'District_Name_SHEOHAR': [0], 'District_Name_SHEOPUR': [0], 'District_Name_SHIMLA': [0], 'District_Name_SHIMOGA': [0], 'District_Name_SHIVPURI': [0], 
'District_Name_SHOPIAN': [0], 'District_Name_SHRAVASTI': [0], 'District_Name_SIDDHARTH NAGAR': [0], 'District_Name_SIDHI': [0], 'District_Name_SIKAR': [0], 'District_Name_SIMDEGA': [0], 'District_Name_SINDHUDURG': [0], 'District_Name_SINGRAULI': [0], 'District_Name_SIRMAUR': [0], 'District_Name_SIROHI': [0], 'District_Name_SIRSA': [0], 'District_Name_SITAMARHI': [0], 'District_Name_SITAPUR': [0], 'District_Name_SIVAGANGA': [0], 'District_Name_SIVASAGAR': [0], 'District_Name_SIWAN': [0], 'District_Name_SOLAN': [0], 'District_Name_SOLAPUR': [0], 'District_Name_SONBHADRA': [0], 'District_Name_SONEPUR': [0], 'District_Name_SONIPAT': [0], 'District_Name_SONITPUR': [0], 'District_Name_SOUTH ANDAMANS': [0], 'District_Name_SOUTH DISTRICT': [0], 'District_Name_SOUTH GARO HILLS': [0], 'District_Name_SOUTH GOA': [0], 'District_Name_SOUTH TRIPURA': [0], 'District_Name_SOUTH WEST GARO HILLS': [0], 'District_Name_SOUTH WEST KHASI HILLS': [0], 'District_Name_SPSR NELLORE': [0], 'District_Name_SRIKAKULAM': [0], 'District_Name_SRINAGAR': [0], 'District_Name_SUKMA': [0], 'District_Name_SULTANPUR': [0], 'District_Name_SUNDARGARH': [0], 'District_Name_SUPAUL': [0], 'District_Name_SURAJPUR': [0], 'District_Name_SURAT': [0], 'District_Name_SURENDRANAGAR': [0], 'District_Name_SURGUJA': [0], 'District_Name_TAMENGLONG': [0], 'District_Name_TAPI': [0], 'District_Name_TARN TARAN': [0], 'District_Name_TAWANG': [0], 'District_Name_TEHRI GARHWAL': [0], 'District_Name_THANE': [0], 'District_Name_THANJAVUR': [0], 'District_Name_THE NILGIRIS': [0], 'District_Name_THENI': [0], 'District_Name_THIRUVALLUR': [0], 'District_Name_THIRUVANANTHAPURAM': [0], 'District_Name_THIRUVARUR': [0], 'District_Name_THOUBAL': [0], 'District_Name_THRISSUR': [0], 'District_Name_TIKAMGARH': [0], 'District_Name_TINSUKIA': [0], 'District_Name_TIRAP': [0], 'District_Name_TIRUCHIRAPPALLI': [0], 'District_Name_TIRUNELVELI': [0], 'District_Name_TIRUPPUR': [0], 'District_Name_TIRUVANNAMALAI': [0], 'District_Name_TONK': [0], 'District_Name_TUENSANG': [0], 'District_Name_TUMKUR': [0], 'District_Name_TUTICORIN': [0], 'District_Name_UDAIPUR': [0], 'District_Name_UDALGURI': [0], 'District_Name_UDAM SINGH NAGAR': [0], 'District_Name_UDHAMPUR': [0], 
'District_Name_UDUPI': [0], 'District_Name_UJJAIN': [0], 'District_Name_UKHRUL': [0], 'District_Name_UMARIA': [0], 'District_Name_UNA': [0], 'District_Name_UNAKOTI': [0], 'District_Name_UNNAO': [0], 'District_Name_UPPER SIANG': [0], 'District_Name_UPPER SUBANSIRI': [0], 'District_Name_UTTAR KANNAD': [0], 'District_Name_UTTAR KASHI': [0], 'District_Name_VADODARA': [0], 'District_Name_VAISHALI': [0], 'District_Name_VALSAD': [0], 'District_Name_VARANASI': [0], 'District_Name_VELLORE': [0], 'District_Name_VIDISHA': [0], 'District_Name_VILLUPURAM': [0], 'District_Name_VIRUDHUNAGAR': [0], 'District_Name_VISAKHAPATANAM': [0], 'District_Name_VIZIANAGARAM': [0], 'District_Name_WARANGAL': [0], 'District_Name_WARDHA': [0], 'District_Name_WASHIM': [0], 'District_Name_WAYANAD': [0], 'District_Name_WEST DISTRICT': [0], 'District_Name_WEST GARO HILLS': [0], 'District_Name_WEST GODAVARI': [0], 'District_Name_WEST JAINTIA HILLS': [0], 'District_Name_WEST KAMENG': [0], 'District_Name_WEST KHASI HILLS': [0], 'District_Name_WEST SIANG': [0], 'District_Name_WEST SINGHBHUM': [0], 'District_Name_WEST TRIPURA': [0], 'District_Name_WOKHA': [0], 'District_Name_YADGIR': [0], 'District_Name_YAMUNANAGAR': [0], 'District_Name_YANAM': [0], 'District_Name_YAVATMAL': [0], 'District_Name_ZUNHEBOTO': [0], 'Season_Autumn     ': [0], 'Season_Kharif     ': [0], 'Season_Rabi       ': [0], 'Season_Summer     ': [0], 'Season_Whole Year ': [0], 'Season_Winter     ': [0], 'Crop_Apple': [0], 'Crop_Arcanut (Processed)': [0], 'Crop_Arecanut': [0], 'Crop_Arhar/Tur': [0], 'Crop_Ash Gourd': [0], 'Crop_Atcanut (Raw)': [0], 'Crop_Bajra': [0], 'Crop_Banana': [0], 'Crop_Barley': [0], 'Crop_Bean': [0], 'Crop_Beans & Mutter(Vegetable)': [0], 'Crop_Beet Root': [0], 'Crop_Ber': [0], 'Crop_Bhindi': [0], 'Crop_Bitter Gourd': [0], 'Crop_Black pepper': [0], 'Crop_Blackgram': [0], 'Crop_Bottle Gourd': [0], 'Crop_Brinjal': [0], 'Crop_Cabbage': [0], 'Crop_Cardamom': [0], 'Crop_Carrot': [0], 'Crop_Cashewnut': [0], 'Crop_Cashewnut Processed': [0], 'Crop_Cashewnut Raw': [0], 'Crop_Castor seed': [0], 'Crop_Cauliflower': [0], 'Crop_Citrus Fruit': [0], 'Crop_Coconut ': [0], 'Crop_Coffee': [0], 'Crop_Colocosia': [0], 'Crop_Cond-spcs other': [0], 'Crop_Coriander': [0], 'Crop_Cotton(lint)': [0], 'Crop_Cowpea(Lobia)': [0], 'Crop_Cucumber': [0], 'Crop_Drum Stick': [0], 'Crop_Dry chillies': [0], 'Crop_Dry ginger': [0], 'Crop_Garlic': [0], 'Crop_Ginger': [0], 'Crop_Gram': [0], 'Crop_Grapes': [0], 'Crop_Groundnut': [0], 'Crop_Guar seed': [0], 'Crop_Horse-gram': [0], 'Crop_Jack Fruit': [0], 'Crop_Jobster': [0], 'Crop_Jowar': [0], 'Crop_Jute': [0], 'Crop_Jute & mesta': [0], 'Crop_Kapas': [0], 'Crop_Khesari': [0], 'Crop_Korra': [0], 'Crop_Lab-Lab': [0], 'Crop_Lemon': [0], 'Crop_Lentil': [0], 'Crop_Linseed': [0], 'Crop_Litchi': [0], 'Crop_Maize': [0], 'Crop_Mango': [0], 'Crop_Masoor': [0], 'Crop_Mesta': [0], 'Crop_Moong(Green Gram)': [0], 'Crop_Moth': [0], 'Crop_Niger seed': [0], 'Crop_Oilseeds total': [0], 'Crop_Onion': [0], 'Crop_Orange': [0], 'Crop_Other  Rabi pulses': [0], 'Crop_Other Cereals & Millets': [0], 'Crop_Other Citrus Fruit': [0], 'Crop_Other Dry Fruit': [0], 'Crop_Other Fresh Fruits': [0], 'Crop_Other Kharif pulses': [0], 'Crop_Other Vegetables': [0], 'Crop_Paddy': [0], 'Crop_Papaya': [0], 'Crop_Peach': [0], 'Crop_Pear': [0], 'Crop_Peas  (vegetable)': [0], 'Crop_Peas & beans (Pulses)': [0], 'Crop_Perilla': [0], 'Crop_Pineapple': [0], 'Crop_Plums': [0], 'Crop_Pome Fruit': [0], 'Crop_Pome Granet': [0], 'Crop_Potato': [0], 'Crop_Pulses total': [0], 'Crop_Pump Kin': [0], 'Crop_Ragi': [0], 'Crop_Rajmash Kholar': [0], 'Crop_Rapeseed &Mustard': [0], 'Crop_Redish': [0], 'Crop_Ribed Guard': [0], 'Crop_Rice': [0], 'Crop_Ricebean (nagadal)': [0], 'Crop_Rubber': [0], 'Crop_Safflower': [0], 'Crop_Samai': [0], 'Crop_Sannhamp': [0], 'Crop_Sapota': [0], 'Crop_Sesamum': [0], 'Crop_Small millets': [0], 'Crop_Snak Guard': [0], 'Crop_Soyabean': [0], 'Crop_Sugarcane': [0], 'Crop_Sunflower': [0], 'Crop_Sweet potato': [0], 'Crop_Tapioca': [0], 'Crop_Tea': [0], 'Crop_Tobacco': [0], 'Crop_Tomato': [0], 'Crop_Total foodgrain': [0], 'Crop_Turmeric': [0], 'Crop_Turnip': [0], 'Crop_Urad': [0], 'Crop_Varagu': [0], 'Crop_Water Melon': [0], 'Crop_Wheat': [0], 'Crop_Yam': [0], 'Crop_other fibres': [0], 'Crop_other misc. pulses': [0], 'Crop_other oilseeds': [0]}

from tkinter import *
from tkinter import ttk
import tkinter as tk
from PIL import ImageTk, Image

root = Tk()
root.title('Crop Recommendation')
root.geometry('1000x800')
root.configure(background="#f0f0f0")

# Load the background image
bg_image = Image.open("bg.png")
bg_image = bg_image.resize((1000, 800), Image.ANTIALIAS)
bg_image = ImageTk.PhotoImage(bg_image)

# Create a label for the background image
bg_label = Label(root, image=bg_image)
bg_label.place(x=0, y=0, relwidth=1, relheight=1)

# Define a consistent font for the UI
font = ('Helvetica', 16)

# Add a consistent color scheme for the UI
bg_color = "#f0f0f0"
fg_color = "#333333"
btn_color = "#4b4b4b"
btn_fg_color = "#ffffff"

# Create a label for the title
title_label = Label(root, text="Crop Recommendation", font=('arial', 20, 'bold'), bg=bg_color, fg=fg_color)
title_label.grid(row=0, column=0, columnspan=2, pady=20)

title2_label = Label(root, text="Yeild Prediction", font=('arial', 20, 'bold'), bg=bg_color, fg=fg_color)
title2_label.grid(row=0, column=3, columnspan=2, pady=20)

# Create labels for the input fields
nitrogen_label = ttk.Label(root, text='Nitrogen:', font=font, background=bg_color, foreground=fg_color)
nitrogen_label.grid(row=1, column=0, padx=10, pady=10)

phosphorus_label = ttk.Label(root, text='Phosphorus:', font=font, background=bg_color, foreground=fg_color)
phosphorus_label.grid(row=2, column=0, padx=10, pady=10)

potassium_label = ttk.Label(root, text='Potassium:', font=font, background=bg_color, foreground=fg_color)
potassium_label.grid(row=3, column=0, padx=10, pady=10)

temperature_label = ttk.Label(root, text='Temperature:(°C)', font=font, background=bg_color, foreground=fg_color)
temperature_label.grid(row=4, column=0, padx=10, pady=10)

humidity_label = ttk.Label(root, text='Humidity:(g/m3)', font=font, background=bg_color, foreground=fg_color)
humidity_label.grid(row=5, column=0, padx=10, pady=10)

ph_label = ttk.Label(root, text='PH:', font=font, background=bg_color, foreground=fg_color)
ph_label.grid(row=6, column=0, padx=10, pady=10)

rainfall_label = ttk.Label(root, text='Rainfall:(mm)', font=font, background=bg_color, foreground=fg_color)
rainfall_label.grid(row=7, column=0, padx=10, pady=10)

# Create input fields
nitrogen_entry = Entry(root, width=30)
nitrogen_entry.grid(row=1, column=1, padx=10, pady=10)

phosphorus_entry = Entry(root, width=30)
phosphorus_entry.grid(row=2, column=1, padx=10, pady=10)

pottasium_entry = Entry(root, width=30)
pottasium_entry.grid(row=3, column=1, padx=10, pady=10)

temperature_entry = Entry(root, width=30)
temperature_entry.grid(row=4, column=1, padx=10, pady=10)

humidity_entry = Entry(root, width=30)
humidity_entry.grid(row=5, column=1, padx=10, pady=10)

ph_entry = Entry(root, width=30)
ph_entry.grid(row=6, column=1, padx=10, pady=10)

rainfall_entry = Entry(root, width=30)
rainfall_entry.grid(row=7, column=1, padx=10, pady=10)
  

def recommend():
        n = nitrogen_entry.get()
        p = phosphorus_entry.get()
        k = pottasium_entry.get()
        temperature = temperature_entry.get()
        humidity = humidity_entry.get()
        ph = ph_entry.get()
        rainfall = rainfall_entry.get()
        data = {'N': [int(n)],
        'P': [int(p)],
        'K': [int(k)],
        'temperature': [float(temperature)],
        'humidity': [float(humidity)],
        'ph': [float(ph)],
        'rainfall': [float(rainfall)]}
        print(data)         

        df = pd.DataFrame(data)
        print(df.head())
        out = np.argmax(multi_target_forest.predict(df)) 
        print(out)
        crop_name.set(final[out])

   
ttk.Button(root, text='Recommendation', command=recommend, style='TButton').grid(row=10, column=0, columnspan=2, pady=20)
    

crop_name = StringVar()
crop_name.set('')

# Create a Label widget to display the predicted crop name
crop_label = Label(root, textvariable=crop_name, font=font, background=bg_color, foreground=fg_color)
crop_label.grid(row=12, column=0, columnspan=2, padx=10, pady=10)

# Create the state label and dropdown
state_label = ttk.Label(root, text='State:', font=font, background=bg_color, foreground=fg_color)
state_label.grid(row=1, column=3, padx=10, pady=10)

state_var = StringVar()
state_dropdown = ttk.OptionMenu(root, state_var, *state_districts.keys())
state_dropdown.grid(row=1, column=4, padx=10, pady=10)

# Create the district label and dropdown
district_label = ttk.Label(root, text='District:', font=font, background=bg_color, foreground=fg_color)
district_label.grid(row=2, column=3, padx=10, pady=10)

district_var = StringVar()
district_dropdown = ttk.OptionMenu(root, district_var, '')
district_dropdown.grid(row=2, column=4, padx=10, pady=10)

# Function to update the district dropdown when a new state is selected
def update_district_dropdown(*args):
    # Get the selected state
    selected_state = state_var.get()

    # Update the district dropdown options based on the selected state
    district_dropdown['menu'].delete(0, 'end')
    for district in state_districts[selected_state]:
        district_dropdown['menu'].add_command(label=district, command=lambda value=district: district_var.set(value))

# Update the district dropdown when a new state is selected
state_var.trace('w', update_district_dropdown)


season_label = ttk.Label(root, text='Season:', font=font, background=bg_color, foreground=fg_color)
season_label.grid(row=3,column=3, padx=10, pady=10)

options_season = StringVar(root)
options_season.set("Select Option") # default value
    
om3 = ttk.OptionMenu(root, options_season, 'Kharif     ', 'Whole Year ', 'Autumn     ', 'Rabi       ', 'Summer     ', 'Winter     ')
om3.grid(row=3, column=4, padx=10, pady=10) 

area_label = ttk.Label(root, text='Area:', font=font, background=bg_color, foreground=fg_color)
area_label.grid(row=4,column=3, padx=10, pady=10)

area_entry = Entry(root, width=30)
area_entry.grid(row=4, column=4, padx=10, pady=10)



def predict():
        crop = 'Crop_' + crop_name.get()
        district = 'District_Name_' + district_var.get()
        area = area_entry.get()
        season = 'Season_' + options_season.get()
        # model.predict()
        data[crop] = [1]
        data[district] = [1]
        data[season] = [1]
        data["Area"] = [float(area)]
        print(data[crop]," ",data[district]," ",data[season]," ",data["Area"])
        df = pd.DataFrame(data)
        print(df.shape)
        out = model.predict(df)
        out = str(out) + ' Quintals/Acres'
      #   out = float(out)

        Yeild_value.set(out)
        data[crop] = [0]
        data[district] = [0]
        data[season] = [0]
        data["Area"] = [0]
      #   print(data[crop]," ",data[district]," ",data[season]," ",data["Area"])
      #   print(data)
ttk.Button(root, text='Yield Prediction', command=predict, style='TButton').grid(row=5, column=3, columnspan=2,padx=150, pady=10)


Yeild_value = StringVar()
Yeild_value.set('             ')

# Create a Label widget to display the predicted yeild value
yeild_label = Label(root, textvariable=Yeild_value, font=font, background=bg_color, foreground=fg_color)
yeild_label.grid(row=6, column=3, columnspan=10, padx=10, pady=10)

    
root.mainloop()



