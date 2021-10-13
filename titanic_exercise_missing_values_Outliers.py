{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 69,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 70,
   "metadata": {},
   "outputs": [],
   "source": [
    "import ssl\n",
    "ssl._create_default_https_context = ssl._create_unverified_context"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 71,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = sns.load_dataset(\"titanic\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 72,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>survived</th>\n",
       "      <th>pclass</th>\n",
       "      <th>sex</th>\n",
       "      <th>age</th>\n",
       "      <th>sibsp</th>\n",
       "      <th>parch</th>\n",
       "      <th>fare</th>\n",
       "      <th>embarked</th>\n",
       "      <th>class</th>\n",
       "      <th>who</th>\n",
       "      <th>adult_male</th>\n",
       "      <th>deck</th>\n",
       "      <th>embark_town</th>\n",
       "      <th>alive</th>\n",
       "      <th>alone</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>male</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>7.2500</td>\n",
       "      <td>S</td>\n",
       "      <td>Third</td>\n",
       "      <td>man</td>\n",
       "      <td>True</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Southampton</td>\n",
       "      <td>no</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>female</td>\n",
       "      <td>38.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>71.2833</td>\n",
       "      <td>C</td>\n",
       "      <td>First</td>\n",
       "      <td>woman</td>\n",
       "      <td>False</td>\n",
       "      <td>C</td>\n",
       "      <td>Cherbourg</td>\n",
       "      <td>yes</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>female</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7.9250</td>\n",
       "      <td>S</td>\n",
       "      <td>Third</td>\n",
       "      <td>woman</td>\n",
       "      <td>False</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Southampton</td>\n",
       "      <td>yes</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>female</td>\n",
       "      <td>35.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>53.1000</td>\n",
       "      <td>S</td>\n",
       "      <td>First</td>\n",
       "      <td>woman</td>\n",
       "      <td>False</td>\n",
       "      <td>C</td>\n",
       "      <td>Southampton</td>\n",
       "      <td>yes</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>male</td>\n",
       "      <td>35.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>S</td>\n",
       "      <td>Third</td>\n",
       "      <td>man</td>\n",
       "      <td>True</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Southampton</td>\n",
       "      <td>no</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   survived  pclass     sex   age  sibsp  parch     fare embarked  class  \\\n",
       "0         0       3    male  22.0      1      0   7.2500        S  Third   \n",
       "1         1       1  female  38.0      1      0  71.2833        C  First   \n",
       "2         1       3  female  26.0      0      0   7.9250        S  Third   \n",
       "3         1       1  female  35.0      1      0  53.1000        S  First   \n",
       "4         0       3    male  35.0      0      0   8.0500        S  Third   \n",
       "\n",
       "     who  adult_male deck  embark_town alive  alone  \n",
       "0    man        True  NaN  Southampton    no  False  \n",
       "1  woman       False    C    Cherbourg   yes  False  \n",
       "2  woman       False  NaN  Southampton   yes   True  \n",
       "3  woman       False    C  Southampton   yes  False  \n",
       "4    man        True  NaN  Southampton    no   True  "
      ]
     },
     "execution_count": 72,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 73,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 891 entries, 0 to 890\n",
      "Data columns (total 15 columns):\n",
      "survived       891 non-null int64\n",
      "pclass         891 non-null int64\n",
      "sex            891 non-null object\n",
      "age            714 non-null float64\n",
      "sibsp          891 non-null int64\n",
      "parch          891 non-null int64\n",
      "fare           891 non-null float64\n",
      "embarked       889 non-null object\n",
      "class          891 non-null category\n",
      "who            891 non-null object\n",
      "adult_male     891 non-null bool\n",
      "deck           203 non-null category\n",
      "embark_town    889 non-null object\n",
      "alive          891 non-null object\n",
      "alone          891 non-null bool\n",
      "dtypes: bool(2), category(2), float64(2), int64(4), object(5)\n",
      "memory usage: 80.6+ KB\n"
     ]
    }
   ],
   "source": [
    "df.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 74,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "survived         0\n",
       "pclass           0\n",
       "sex              0\n",
       "age            177\n",
       "sibsp            0\n",
       "parch            0\n",
       "fare             0\n",
       "embarked         2\n",
       "class            0\n",
       "who              0\n",
       "adult_male       0\n",
       "deck           688\n",
       "embark_town      2\n",
       "alive            0\n",
       "alone            0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 74,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 75,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(891, 15)"
      ]
     },
     "execution_count": 75,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 76,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>survived</th>\n",
       "      <th>pclass</th>\n",
       "      <th>age</th>\n",
       "      <th>sibsp</th>\n",
       "      <th>parch</th>\n",
       "      <th>fare</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>891.000000</td>\n",
       "      <td>891.000000</td>\n",
       "      <td>714.000000</td>\n",
       "      <td>891.000000</td>\n",
       "      <td>891.000000</td>\n",
       "      <td>891.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>0.383838</td>\n",
       "      <td>2.308642</td>\n",
       "      <td>29.699118</td>\n",
       "      <td>0.523008</td>\n",
       "      <td>0.381594</td>\n",
       "      <td>32.204208</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>0.486592</td>\n",
       "      <td>0.836071</td>\n",
       "      <td>14.526497</td>\n",
       "      <td>1.102743</td>\n",
       "      <td>0.806057</td>\n",
       "      <td>49.693429</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.420000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>0.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>20.125000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>7.910400</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>0.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>28.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>14.454200</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>1.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>38.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>31.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>1.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>80.000000</td>\n",
       "      <td>8.000000</td>\n",
       "      <td>6.000000</td>\n",
       "      <td>512.329200</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "         survived      pclass         age       sibsp       parch        fare\n",
       "count  891.000000  891.000000  714.000000  891.000000  891.000000  891.000000\n",
       "mean     0.383838    2.308642   29.699118    0.523008    0.381594   32.204208\n",
       "std      0.486592    0.836071   14.526497    1.102743    0.806057   49.693429\n",
       "min      0.000000    1.000000    0.420000    0.000000    0.000000    0.000000\n",
       "25%      0.000000    2.000000   20.125000    0.000000    0.000000    7.910400\n",
       "50%      0.000000    3.000000   28.000000    0.000000    0.000000   14.454200\n",
       "75%      1.000000    3.000000   38.000000    1.000000    0.000000   31.000000\n",
       "max      1.000000    3.000000   80.000000    8.000000    6.000000  512.329200"
      ]
     },
     "execution_count": 76,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.describe() # descriptive statistics"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 77,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>sex</th>\n",
       "      <th>embarked</th>\n",
       "      <th>class</th>\n",
       "      <th>deck</th>\n",
       "      <th>who</th>\n",
       "      <th>adult_male</th>\n",
       "      <th>embark_town</th>\n",
       "      <th>alive</th>\n",
       "      <th>alone</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>891</td>\n",
       "      <td>889</td>\n",
       "      <td>891</td>\n",
       "      <td>203</td>\n",
       "      <td>891</td>\n",
       "      <td>891</td>\n",
       "      <td>889</td>\n",
       "      <td>891</td>\n",
       "      <td>891</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>unique</th>\n",
       "      <td>2</td>\n",
       "      <td>3</td>\n",
       "      <td>3</td>\n",
       "      <td>7</td>\n",
       "      <td>3</td>\n",
       "      <td>2</td>\n",
       "      <td>3</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>top</th>\n",
       "      <td>male</td>\n",
       "      <td>S</td>\n",
       "      <td>Third</td>\n",
       "      <td>C</td>\n",
       "      <td>man</td>\n",
       "      <td>True</td>\n",
       "      <td>Southampton</td>\n",
       "      <td>no</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>freq</th>\n",
       "      <td>577</td>\n",
       "      <td>644</td>\n",
       "      <td>491</td>\n",
       "      <td>59</td>\n",
       "      <td>537</td>\n",
       "      <td>537</td>\n",
       "      <td>644</td>\n",
       "      <td>549</td>\n",
       "      <td>537</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "         sex embarked  class deck  who adult_male  embark_town alive alone\n",
       "count    891      889    891  203  891        891          889   891   891\n",
       "unique     2        3      3    7    3          2            3     2     2\n",
       "top     male        S  Third    C  man       True  Southampton    no  True\n",
       "freq     577      644    491   59  537        537          644   549   537"
      ]
     },
     "execution_count": 77,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df[[\"sex\", \"embarked\", \"class\", \"deck\", \"who\", \"adult_male\", \"embark_town\", \"alive\", \"alone\"]].describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 78,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "NaN    688\n",
       "C       59\n",
       "B       47\n",
       "D       33\n",
       "E       32\n",
       "A       15\n",
       "F       13\n",
       "G        4\n",
       "Name: deck, dtype: int64"
      ]
     },
     "execution_count": 78,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df[\"deck\"].value_counts(dropna=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 79,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>count</th>\n",
       "      <th>mean</th>\n",
       "      <th>std</th>\n",
       "      <th>min</th>\n",
       "      <th>25%</th>\n",
       "      <th>50%</th>\n",
       "      <th>75%</th>\n",
       "      <th>max</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>deck</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>A</th>\n",
       "      <td>15.0</td>\n",
       "      <td>39.623887</td>\n",
       "      <td>17.975333</td>\n",
       "      <td>0.0000</td>\n",
       "      <td>30.8479</td>\n",
       "      <td>35.50000</td>\n",
       "      <td>50.24790</td>\n",
       "      <td>81.8583</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>B</th>\n",
       "      <td>47.0</td>\n",
       "      <td>113.505764</td>\n",
       "      <td>109.301500</td>\n",
       "      <td>0.0000</td>\n",
       "      <td>57.0000</td>\n",
       "      <td>80.00000</td>\n",
       "      <td>120.00000</td>\n",
       "      <td>512.3292</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>C</th>\n",
       "      <td>59.0</td>\n",
       "      <td>100.151341</td>\n",
       "      <td>70.225588</td>\n",
       "      <td>26.5500</td>\n",
       "      <td>42.5021</td>\n",
       "      <td>83.47500</td>\n",
       "      <td>143.59165</td>\n",
       "      <td>263.0000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>D</th>\n",
       "      <td>33.0</td>\n",
       "      <td>57.244576</td>\n",
       "      <td>29.592832</td>\n",
       "      <td>12.8750</td>\n",
       "      <td>30.0000</td>\n",
       "      <td>53.10000</td>\n",
       "      <td>77.28750</td>\n",
       "      <td>113.2750</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>E</th>\n",
       "      <td>32.0</td>\n",
       "      <td>46.026694</td>\n",
       "      <td>32.608315</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>26.1125</td>\n",
       "      <td>45.18125</td>\n",
       "      <td>56.15730</td>\n",
       "      <td>134.5000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>F</th>\n",
       "      <td>13.0</td>\n",
       "      <td>18.696792</td>\n",
       "      <td>11.728217</td>\n",
       "      <td>7.6500</td>\n",
       "      <td>7.7500</td>\n",
       "      <td>13.00000</td>\n",
       "      <td>26.00000</td>\n",
       "      <td>39.0000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>G</th>\n",
       "      <td>4.0</td>\n",
       "      <td>13.581250</td>\n",
       "      <td>3.601222</td>\n",
       "      <td>10.4625</td>\n",
       "      <td>10.4625</td>\n",
       "      <td>13.58125</td>\n",
       "      <td>16.70000</td>\n",
       "      <td>16.7000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "      count        mean         std      min      25%       50%        75%  \\\n",
       "deck                                                                         \n",
       "A      15.0   39.623887   17.975333   0.0000  30.8479  35.50000   50.24790   \n",
       "B      47.0  113.505764  109.301500   0.0000  57.0000  80.00000  120.00000   \n",
       "C      59.0  100.151341   70.225588  26.5500  42.5021  83.47500  143.59165   \n",
       "D      33.0   57.244576   29.592832  12.8750  30.0000  53.10000   77.28750   \n",
       "E      32.0   46.026694   32.608315   8.0500  26.1125  45.18125   56.15730   \n",
       "F      13.0   18.696792   11.728217   7.6500   7.7500  13.00000   26.00000   \n",
       "G       4.0   13.581250    3.601222  10.4625  10.4625  13.58125   16.70000   \n",
       "\n",
       "           max  \n",
       "deck            \n",
       "A      81.8583  \n",
       "B     512.3292  \n",
       "C     263.0000  \n",
       "D     113.2750  \n",
       "E     134.5000  \n",
       "F      39.0000  \n",
       "G      16.7000  "
      ]
     },
     "execution_count": 79,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.groupby(\"deck\")[\"fare\"].describe()\n",
    "# min \"A\", \"B\" deck fare is zero. That's not possible. we should fill them.\n",
    "\n",
    "# And as we understand from the data \"fare\" is related with \"deck\". so we should focus on these two variables."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 80,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>count</th>\n",
       "      <th>mean</th>\n",
       "      <th>std</th>\n",
       "      <th>min</th>\n",
       "      <th>25%</th>\n",
       "      <th>50%</th>\n",
       "      <th>75%</th>\n",
       "      <th>max</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>deck</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>G</th>\n",
       "      <td>4.0</td>\n",
       "      <td>13.581250</td>\n",
       "      <td>3.601222</td>\n",
       "      <td>10.4625</td>\n",
       "      <td>10.4625</td>\n",
       "      <td>13.58125</td>\n",
       "      <td>16.70000</td>\n",
       "      <td>16.7000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>F</th>\n",
       "      <td>13.0</td>\n",
       "      <td>18.696792</td>\n",
       "      <td>11.728217</td>\n",
       "      <td>7.6500</td>\n",
       "      <td>7.7500</td>\n",
       "      <td>13.00000</td>\n",
       "      <td>26.00000</td>\n",
       "      <td>39.0000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>A</th>\n",
       "      <td>15.0</td>\n",
       "      <td>39.623887</td>\n",
       "      <td>17.975333</td>\n",
       "      <td>0.0000</td>\n",
       "      <td>30.8479</td>\n",
       "      <td>35.50000</td>\n",
       "      <td>50.24790</td>\n",
       "      <td>81.8583</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>E</th>\n",
       "      <td>32.0</td>\n",
       "      <td>46.026694</td>\n",
       "      <td>32.608315</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>26.1125</td>\n",
       "      <td>45.18125</td>\n",
       "      <td>56.15730</td>\n",
       "      <td>134.5000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>D</th>\n",
       "      <td>33.0</td>\n",
       "      <td>57.244576</td>\n",
       "      <td>29.592832</td>\n",
       "      <td>12.8750</td>\n",
       "      <td>30.0000</td>\n",
       "      <td>53.10000</td>\n",
       "      <td>77.28750</td>\n",
       "      <td>113.2750</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>C</th>\n",
       "      <td>59.0</td>\n",
       "      <td>100.151341</td>\n",
       "      <td>70.225588</td>\n",
       "      <td>26.5500</td>\n",
       "      <td>42.5021</td>\n",
       "      <td>83.47500</td>\n",
       "      <td>143.59165</td>\n",
       "      <td>263.0000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>B</th>\n",
       "      <td>47.0</td>\n",
       "      <td>113.505764</td>\n",
       "      <td>109.301500</td>\n",
       "      <td>0.0000</td>\n",
       "      <td>57.0000</td>\n",
       "      <td>80.00000</td>\n",
       "      <td>120.00000</td>\n",
       "      <td>512.3292</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "      count        mean         std      min      25%       50%        75%  \\\n",
       "deck                                                                         \n",
       "G       4.0   13.581250    3.601222  10.4625  10.4625  13.58125   16.70000   \n",
       "F      13.0   18.696792   11.728217   7.6500   7.7500  13.00000   26.00000   \n",
       "A      15.0   39.623887   17.975333   0.0000  30.8479  35.50000   50.24790   \n",
       "E      32.0   46.026694   32.608315   8.0500  26.1125  45.18125   56.15730   \n",
       "D      33.0   57.244576   29.592832  12.8750  30.0000  53.10000   77.28750   \n",
       "C      59.0  100.151341   70.225588  26.5500  42.5021  83.47500  143.59165   \n",
       "B      47.0  113.505764  109.301500   0.0000  57.0000  80.00000  120.00000   \n",
       "\n",
       "           max  \n",
       "deck            \n",
       "G      16.7000  \n",
       "F      39.0000  \n",
       "A      81.8583  \n",
       "E     134.5000  \n",
       "D     113.2750  \n",
       "C     263.0000  \n",
       "B     512.3292  "
      ]
     },
     "execution_count": 80,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.groupby(\"deck\")[\"fare\"].describe().sort_values(by=\"mean\")\n",
    "# minimum mean \"fare\" is \"F\" deck"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 81,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "NaN    327\n",
       "F        4\n",
       "B        3\n",
       "E        1\n",
       "A        1\n",
       "G        0\n",
       "D        0\n",
       "C        0\n",
       "Name: deck, dtype: int64"
      ]
     },
     "execution_count": 81,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df[df[\"fare\"] < 10][\"deck\"].value_counts(dropna=False)\n",
    "# most frequent deck is \"F\""
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### let's fill 327 NaN deck value with \"F\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 82,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Int64Index([  0,   2,   4,   5,  12,  14,  19,  22,  26,  28,\n",
       "            ...\n",
       "            868, 870, 873, 875, 876, 877, 878, 881, 884, 890],\n",
       "           dtype='int64', length=327)"
      ]
     },
     "execution_count": 82,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df[(df[\"fare\"] < 10) & (df[\"deck\"].isnull())].index"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 83,
   "metadata": {},
   "outputs": [],
   "source": [
    "df.loc[df[(df[\"fare\"] < 10) & (df[\"deck\"].isnull())].index, \"deck\"] = \"F\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 84,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "F    331\n",
       "B      3\n",
       "E      1\n",
       "A      1\n",
       "G      0\n",
       "D      0\n",
       "C      0\n",
       "Name: deck, dtype: int64"
      ]
     },
     "execution_count": 84,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df[df[\"fare\"] < 10][\"deck\"].value_counts(dropna=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 85,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>count</th>\n",
       "      <th>mean</th>\n",
       "      <th>std</th>\n",
       "      <th>min</th>\n",
       "      <th>25%</th>\n",
       "      <th>50%</th>\n",
       "      <th>75%</th>\n",
       "      <th>max</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>deck</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>A</th>\n",
       "      <td>15.0</td>\n",
       "      <td>39.623887</td>\n",
       "      <td>17.975333</td>\n",
       "      <td>0.0000</td>\n",
       "      <td>30.8479</td>\n",
       "      <td>35.50000</td>\n",
       "      <td>50.24790</td>\n",
       "      <td>81.8583</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>B</th>\n",
       "      <td>47.0</td>\n",
       "      <td>113.505764</td>\n",
       "      <td>109.301500</td>\n",
       "      <td>0.0000</td>\n",
       "      <td>57.0000</td>\n",
       "      <td>80.00000</td>\n",
       "      <td>120.00000</td>\n",
       "      <td>512.3292</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>C</th>\n",
       "      <td>59.0</td>\n",
       "      <td>100.151341</td>\n",
       "      <td>70.225588</td>\n",
       "      <td>26.5500</td>\n",
       "      <td>42.5021</td>\n",
       "      <td>83.47500</td>\n",
       "      <td>143.59165</td>\n",
       "      <td>263.0000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>D</th>\n",
       "      <td>33.0</td>\n",
       "      <td>57.244576</td>\n",
       "      <td>29.592832</td>\n",
       "      <td>12.8750</td>\n",
       "      <td>30.0000</td>\n",
       "      <td>53.10000</td>\n",
       "      <td>77.28750</td>\n",
       "      <td>113.2750</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>E</th>\n",
       "      <td>32.0</td>\n",
       "      <td>46.026694</td>\n",
       "      <td>32.608315</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>26.1125</td>\n",
       "      <td>45.18125</td>\n",
       "      <td>56.15730</td>\n",
       "      <td>134.5000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>F</th>\n",
       "      <td>340.0</td>\n",
       "      <td>8.012731</td>\n",
       "      <td>3.456975</td>\n",
       "      <td>0.0000</td>\n",
       "      <td>7.5500</td>\n",
       "      <td>7.85420</td>\n",
       "      <td>8.05000</td>\n",
       "      <td>39.0000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>G</th>\n",
       "      <td>4.0</td>\n",
       "      <td>13.581250</td>\n",
       "      <td>3.601222</td>\n",
       "      <td>10.4625</td>\n",
       "      <td>10.4625</td>\n",
       "      <td>13.58125</td>\n",
       "      <td>16.70000</td>\n",
       "      <td>16.7000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "      count        mean         std      min      25%       50%        75%  \\\n",
       "deck                                                                         \n",
       "A      15.0   39.623887   17.975333   0.0000  30.8479  35.50000   50.24790   \n",
       "B      47.0  113.505764  109.301500   0.0000  57.0000  80.00000  120.00000   \n",
       "C      59.0  100.151341   70.225588  26.5500  42.5021  83.47500  143.59165   \n",
       "D      33.0   57.244576   29.592832  12.8750  30.0000  53.10000   77.28750   \n",
       "E      32.0   46.026694   32.608315   8.0500  26.1125  45.18125   56.15730   \n",
       "F     340.0    8.012731    3.456975   0.0000   7.5500   7.85420    8.05000   \n",
       "G       4.0   13.581250    3.601222  10.4625  10.4625  13.58125   16.70000   \n",
       "\n",
       "           max  \n",
       "deck            \n",
       "A      81.8583  \n",
       "B     512.3292  \n",
       "C     263.0000  \n",
       "D     113.2750  \n",
       "E     134.5000  \n",
       "F      39.0000  \n",
       "G      16.7000  "
      ]
     },
     "execution_count": 85,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.groupby(\"deck\")[\"fare\"].describe()\n",
    "# min \"A\", \"B\", \"F\" deck fare is zero. That's not possible. we should fill them."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 86,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>survived</th>\n",
       "      <th>pclass</th>\n",
       "      <th>sex</th>\n",
       "      <th>age</th>\n",
       "      <th>sibsp</th>\n",
       "      <th>parch</th>\n",
       "      <th>fare</th>\n",
       "      <th>embarked</th>\n",
       "      <th>class</th>\n",
       "      <th>who</th>\n",
       "      <th>adult_male</th>\n",
       "      <th>deck</th>\n",
       "      <th>embark_town</th>\n",
       "      <th>alive</th>\n",
       "      <th>alone</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>179</th>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>male</td>\n",
       "      <td>36.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>S</td>\n",
       "      <td>Third</td>\n",
       "      <td>man</td>\n",
       "      <td>True</td>\n",
       "      <td>F</td>\n",
       "      <td>Southampton</td>\n",
       "      <td>no</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>271</th>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>male</td>\n",
       "      <td>25.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>S</td>\n",
       "      <td>Third</td>\n",
       "      <td>man</td>\n",
       "      <td>True</td>\n",
       "      <td>F</td>\n",
       "      <td>Southampton</td>\n",
       "      <td>yes</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>277</th>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>male</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>S</td>\n",
       "      <td>Second</td>\n",
       "      <td>man</td>\n",
       "      <td>True</td>\n",
       "      <td>F</td>\n",
       "      <td>Southampton</td>\n",
       "      <td>no</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>302</th>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>male</td>\n",
       "      <td>19.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>S</td>\n",
       "      <td>Third</td>\n",
       "      <td>man</td>\n",
       "      <td>True</td>\n",
       "      <td>F</td>\n",
       "      <td>Southampton</td>\n",
       "      <td>no</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>413</th>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>male</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>S</td>\n",
       "      <td>Second</td>\n",
       "      <td>man</td>\n",
       "      <td>True</td>\n",
       "      <td>F</td>\n",
       "      <td>Southampton</td>\n",
       "      <td>no</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>466</th>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>male</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>S</td>\n",
       "      <td>Second</td>\n",
       "      <td>man</td>\n",
       "      <td>True</td>\n",
       "      <td>F</td>\n",
       "      <td>Southampton</td>\n",
       "      <td>no</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>481</th>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>male</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>S</td>\n",
       "      <td>Second</td>\n",
       "      <td>man</td>\n",
       "      <td>True</td>\n",
       "      <td>F</td>\n",
       "      <td>Southampton</td>\n",
       "      <td>no</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>597</th>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>male</td>\n",
       "      <td>49.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>S</td>\n",
       "      <td>Third</td>\n",
       "      <td>man</td>\n",
       "      <td>True</td>\n",
       "      <td>F</td>\n",
       "      <td>Southampton</td>\n",
       "      <td>no</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>633</th>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>male</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>S</td>\n",
       "      <td>First</td>\n",
       "      <td>man</td>\n",
       "      <td>True</td>\n",
       "      <td>F</td>\n",
       "      <td>Southampton</td>\n",
       "      <td>no</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>674</th>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>male</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>S</td>\n",
       "      <td>Second</td>\n",
       "      <td>man</td>\n",
       "      <td>True</td>\n",
       "      <td>F</td>\n",
       "      <td>Southampton</td>\n",
       "      <td>no</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>732</th>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>male</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>S</td>\n",
       "      <td>Second</td>\n",
       "      <td>man</td>\n",
       "      <td>True</td>\n",
       "      <td>F</td>\n",
       "      <td>Southampton</td>\n",
       "      <td>no</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>822</th>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>male</td>\n",
       "      <td>38.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>S</td>\n",
       "      <td>First</td>\n",
       "      <td>man</td>\n",
       "      <td>True</td>\n",
       "      <td>F</td>\n",
       "      <td>Southampton</td>\n",
       "      <td>no</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "     survived  pclass   sex   age  sibsp  parch  fare embarked   class  who  \\\n",
       "179         0       3  male  36.0      0      0   0.0        S   Third  man   \n",
       "271         1       3  male  25.0      0      0   0.0        S   Third  man   \n",
       "277         0       2  male   NaN      0      0   0.0        S  Second  man   \n",
       "302         0       3  male  19.0      0      0   0.0        S   Third  man   \n",
       "413         0       2  male   NaN      0      0   0.0        S  Second  man   \n",
       "466         0       2  male   NaN      0      0   0.0        S  Second  man   \n",
       "481         0       2  male   NaN      0      0   0.0        S  Second  man   \n",
       "597         0       3  male  49.0      0      0   0.0        S   Third  man   \n",
       "633         0       1  male   NaN      0      0   0.0        S   First  man   \n",
       "674         0       2  male   NaN      0      0   0.0        S  Second  man   \n",
       "732         0       2  male   NaN      0      0   0.0        S  Second  man   \n",
       "822         0       1  male  38.0      0      0   0.0        S   First  man   \n",
       "\n",
       "     adult_male deck  embark_town alive  alone  \n",
       "179        True    F  Southampton    no   True  \n",
       "271        True    F  Southampton   yes   True  \n",
       "277        True    F  Southampton    no   True  \n",
       "302        True    F  Southampton    no   True  \n",
       "413        True    F  Southampton    no   True  \n",
       "466        True    F  Southampton    no   True  \n",
       "481        True    F  Southampton    no   True  \n",
       "597        True    F  Southampton    no   True  \n",
       "633        True    F  Southampton    no   True  \n",
       "674        True    F  Southampton    no   True  \n",
       "732        True    F  Southampton    no   True  \n",
       "822        True    F  Southampton    no   True  "
      ]
     },
     "execution_count": 86,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df[(df[\"fare\"]==0) & (df[\"deck\"]==\"F\")]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 87,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "7.8542"
      ]
     },
     "execution_count": 87,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "med_F = df.groupby(\"deck\")[\"fare\"].describe()[\"50%\"][\"F\"]\n",
    "med_F\n",
    "# we will fill \"fare\" for \"deck F\" with its median value"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 88,
   "metadata": {},
   "outputs": [],
   "source": [
    "df.loc[df[(df[\"fare\"]==0) & (df[\"deck\"]==\"F\")].index, \"fare\"] = med_F"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 89,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>survived</th>\n",
       "      <th>pclass</th>\n",
       "      <th>sex</th>\n",
       "      <th>age</th>\n",
       "      <th>sibsp</th>\n",
       "      <th>parch</th>\n",
       "      <th>fare</th>\n",
       "      <th>embarked</th>\n",
       "      <th>class</th>\n",
       "      <th>who</th>\n",
       "      <th>adult_male</th>\n",
       "      <th>deck</th>\n",
       "      <th>embark_town</th>\n",
       "      <th>alive</th>\n",
       "      <th>alone</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "Empty DataFrame\n",
       "Columns: [survived, pclass, sex, age, sibsp, parch, fare, embarked, class, who, adult_male, deck, embark_town, alive, alone]\n",
       "Index: []"
      ]
     },
     "execution_count": 89,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df[(df[\"fare\"]==0) & (df[\"deck\"]==\"F\")]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 90,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>count</th>\n",
       "      <th>mean</th>\n",
       "      <th>std</th>\n",
       "      <th>min</th>\n",
       "      <th>25%</th>\n",
       "      <th>50%</th>\n",
       "      <th>75%</th>\n",
       "      <th>max</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>deck</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>A</th>\n",
       "      <td>15.0</td>\n",
       "      <td>39.623887</td>\n",
       "      <td>17.975333</td>\n",
       "      <td>0.0000</td>\n",
       "      <td>30.8479</td>\n",
       "      <td>35.50000</td>\n",
       "      <td>50.24790</td>\n",
       "      <td>81.8583</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>B</th>\n",
       "      <td>47.0</td>\n",
       "      <td>113.505764</td>\n",
       "      <td>109.301500</td>\n",
       "      <td>0.0000</td>\n",
       "      <td>57.0000</td>\n",
       "      <td>80.00000</td>\n",
       "      <td>120.00000</td>\n",
       "      <td>512.3292</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>C</th>\n",
       "      <td>59.0</td>\n",
       "      <td>100.151341</td>\n",
       "      <td>70.225588</td>\n",
       "      <td>26.5500</td>\n",
       "      <td>42.5021</td>\n",
       "      <td>83.47500</td>\n",
       "      <td>143.59165</td>\n",
       "      <td>263.0000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>D</th>\n",
       "      <td>33.0</td>\n",
       "      <td>57.244576</td>\n",
       "      <td>29.592832</td>\n",
       "      <td>12.8750</td>\n",
       "      <td>30.0000</td>\n",
       "      <td>53.10000</td>\n",
       "      <td>77.28750</td>\n",
       "      <td>113.2750</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>E</th>\n",
       "      <td>32.0</td>\n",
       "      <td>46.026694</td>\n",
       "      <td>32.608315</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>26.1125</td>\n",
       "      <td>45.18125</td>\n",
       "      <td>56.15730</td>\n",
       "      <td>134.5000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>F</th>\n",
       "      <td>340.0</td>\n",
       "      <td>8.289938</td>\n",
       "      <td>3.098676</td>\n",
       "      <td>4.0125</td>\n",
       "      <td>7.7333</td>\n",
       "      <td>7.85420</td>\n",
       "      <td>8.05000</td>\n",
       "      <td>39.0000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>G</th>\n",
       "      <td>4.0</td>\n",
       "      <td>13.581250</td>\n",
       "      <td>3.601222</td>\n",
       "      <td>10.4625</td>\n",
       "      <td>10.4625</td>\n",
       "      <td>13.58125</td>\n",
       "      <td>16.70000</td>\n",
       "      <td>16.7000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "      count        mean         std      min      25%       50%        75%  \\\n",
       "deck                                                                         \n",
       "A      15.0   39.623887   17.975333   0.0000  30.8479  35.50000   50.24790   \n",
       "B      47.0  113.505764  109.301500   0.0000  57.0000  80.00000  120.00000   \n",
       "C      59.0  100.151341   70.225588  26.5500  42.5021  83.47500  143.59165   \n",
       "D      33.0   57.244576   29.592832  12.8750  30.0000  53.10000   77.28750   \n",
       "E      32.0   46.026694   32.608315   8.0500  26.1125  45.18125   56.15730   \n",
       "F     340.0    8.289938    3.098676   4.0125   7.7333   7.85420    8.05000   \n",
       "G       4.0   13.581250    3.601222  10.4625  10.4625  13.58125   16.70000   \n",
       "\n",
       "           max  \n",
       "deck            \n",
       "A      81.8583  \n",
       "B     512.3292  \n",
       "C     263.0000  \n",
       "D     113.2750  \n",
       "E     134.5000  \n",
       "F      39.0000  \n",
       "G      16.7000  "
      ]
     },
     "execution_count": 90,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.groupby(\"deck\")[\"fare\"].describe()\n",
    "# min \"A\", \"B\", \"F\" deck fare is zero. That's not possible. we should fill them."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 91,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>survived</th>\n",
       "      <th>pclass</th>\n",
       "      <th>sex</th>\n",
       "      <th>age</th>\n",
       "      <th>sibsp</th>\n",
       "      <th>parch</th>\n",
       "      <th>fare</th>\n",
       "      <th>embarked</th>\n",
       "      <th>class</th>\n",
       "      <th>who</th>\n",
       "      <th>adult_male</th>\n",
       "      <th>deck</th>\n",
       "      <th>embark_town</th>\n",
       "      <th>alive</th>\n",
       "      <th>alone</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>263</th>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>male</td>\n",
       "      <td>40.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>S</td>\n",
       "      <td>First</td>\n",
       "      <td>man</td>\n",
       "      <td>True</td>\n",
       "      <td>B</td>\n",
       "      <td>Southampton</td>\n",
       "      <td>no</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>806</th>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>male</td>\n",
       "      <td>39.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>S</td>\n",
       "      <td>First</td>\n",
       "      <td>man</td>\n",
       "      <td>True</td>\n",
       "      <td>A</td>\n",
       "      <td>Southampton</td>\n",
       "      <td>no</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>815</th>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>male</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>S</td>\n",
       "      <td>First</td>\n",
       "      <td>man</td>\n",
       "      <td>True</td>\n",
       "      <td>B</td>\n",
       "      <td>Southampton</td>\n",
       "      <td>no</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "     survived  pclass   sex   age  sibsp  parch  fare embarked  class  who  \\\n",
       "263         0       1  male  40.0      0      0   0.0        S  First  man   \n",
       "806         0       1  male  39.0      0      0   0.0        S  First  man   \n",
       "815         0       1  male   NaN      0      0   0.0        S  First  man   \n",
       "\n",
       "     adult_male deck  embark_town alive  alone  \n",
       "263        True    B  Southampton    no   True  \n",
       "806        True    A  Southampton    no   True  \n",
       "815        True    B  Southampton    no   True  "
      ]
     },
     "execution_count": 91,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df[df[\"fare\"] == 0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 92,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>deck</th>\n",
       "      <th>fare</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>806</th>\n",
       "      <td>A</td>\n",
       "      <td>0.0000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>284</th>\n",
       "      <td>A</td>\n",
       "      <td>26.0000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>630</th>\n",
       "      <td>A</td>\n",
       "      <td>30.0000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>174</th>\n",
       "      <td>A</td>\n",
       "      <td>30.6958</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>209</th>\n",
       "      <td>A</td>\n",
       "      <td>31.0000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>96</th>\n",
       "      <td>A</td>\n",
       "      <td>34.6542</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>23</th>\n",
       "      <td>A</td>\n",
       "      <td>35.5000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>647</th>\n",
       "      <td>A</td>\n",
       "      <td>35.5000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>556</th>\n",
       "      <td>A</td>\n",
       "      <td>39.6000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>583</th>\n",
       "      <td>A</td>\n",
       "      <td>40.1250</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>185</th>\n",
       "      <td>A</td>\n",
       "      <td>50.0000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>867</th>\n",
       "      <td>A</td>\n",
       "      <td>50.4958</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>475</th>\n",
       "      <td>A</td>\n",
       "      <td>52.0000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>599</th>\n",
       "      <td>A</td>\n",
       "      <td>56.9292</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>445</th>\n",
       "      <td>A</td>\n",
       "      <td>81.8583</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "    deck     fare\n",
       "806    A   0.0000\n",
       "284    A  26.0000\n",
       "630    A  30.0000\n",
       "174    A  30.6958\n",
       "209    A  31.0000\n",
       "96     A  34.6542\n",
       "23     A  35.5000\n",
       "647    A  35.5000\n",
       "556    A  39.6000\n",
       "583    A  40.1250\n",
       "185    A  50.0000\n",
       "867    A  50.4958\n",
       "475    A  52.0000\n",
       "599    A  56.9292\n",
       "445    A  81.8583"
      ]
     },
     "execution_count": 92,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df[df[\"deck\"] == \"A\"][[\"deck\",\"fare\"]].sort_values(by=\"fare\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 93,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "35.5"
      ]
     },
     "execution_count": 93,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df[df[\"deck\"]==\"A\"][\"fare\"].median()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 94,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "35.5"
      ]
     },
     "execution_count": 94,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#other method\n",
    "df.groupby(\"deck\").median()[\"fare\"][\"A\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 95,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>survived</th>\n",
       "      <th>pclass</th>\n",
       "      <th>sex</th>\n",
       "      <th>age</th>\n",
       "      <th>sibsp</th>\n",
       "      <th>parch</th>\n",
       "      <th>fare</th>\n",
       "      <th>embarked</th>\n",
       "      <th>class</th>\n",
       "      <th>who</th>\n",
       "      <th>adult_male</th>\n",
       "      <th>deck</th>\n",
       "      <th>embark_town</th>\n",
       "      <th>alive</th>\n",
       "      <th>alone</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>806</th>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>male</td>\n",
       "      <td>39.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>S</td>\n",
       "      <td>First</td>\n",
       "      <td>man</td>\n",
       "      <td>True</td>\n",
       "      <td>A</td>\n",
       "      <td>Southampton</td>\n",
       "      <td>no</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "     survived  pclass   sex   age  sibsp  parch  fare embarked  class  who  \\\n",
       "806         0       1  male  39.0      0      0   0.0        S  First  man   \n",
       "\n",
       "     adult_male deck  embark_town alive  alone  \n",
       "806        True    A  Southampton    no   True  "
      ]
     },
     "execution_count": 95,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df[(df[\"deck\"]==\"A\") & (df[\"fare\"]==0)]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 96,
   "metadata": {},
   "outputs": [],
   "source": [
    "df.loc[df[(df[\"deck\"]==\"A\") & (df[\"fare\"]==0)].index, \"fare\"] = df[df[\"deck\"]==\"A\"][\"fare\"].median()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 97,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>survived</th>\n",
       "      <th>pclass</th>\n",
       "      <th>sex</th>\n",
       "      <th>age</th>\n",
       "      <th>sibsp</th>\n",
       "      <th>parch</th>\n",
       "      <th>fare</th>\n",
       "      <th>embarked</th>\n",
       "      <th>class</th>\n",
       "      <th>who</th>\n",
       "      <th>adult_male</th>\n",
       "      <th>deck</th>\n",
       "      <th>embark_town</th>\n",
       "      <th>alive</th>\n",
       "      <th>alone</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>263</th>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>male</td>\n",
       "      <td>40.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>S</td>\n",
       "      <td>First</td>\n",
       "      <td>man</td>\n",
       "      <td>True</td>\n",
       "      <td>B</td>\n",
       "      <td>Southampton</td>\n",
       "      <td>no</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>815</th>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>male</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>S</td>\n",
       "      <td>First</td>\n",
       "      <td>man</td>\n",
       "      <td>True</td>\n",
       "      <td>B</td>\n",
       "      <td>Southampton</td>\n",
       "      <td>no</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "     survived  pclass   sex   age  sibsp  parch  fare embarked  class  who  \\\n",
       "263         0       1  male  40.0      0      0   0.0        S  First  man   \n",
       "815         0       1  male   NaN      0      0   0.0        S  First  man   \n",
       "\n",
       "     adult_male deck  embark_town alive  alone  \n",
       "263        True    B  Southampton    no   True  \n",
       "815        True    B  Southampton    no   True  "
      ]
     },
     "execution_count": 97,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df[df[\"fare\"] == 0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 98,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>count</th>\n",
       "      <th>mean</th>\n",
       "      <th>std</th>\n",
       "      <th>min</th>\n",
       "      <th>25%</th>\n",
       "      <th>50%</th>\n",
       "      <th>75%</th>\n",
       "      <th>max</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>deck</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>A</th>\n",
       "      <td>15.0</td>\n",
       "      <td>41.990553</td>\n",
       "      <td>14.358954</td>\n",
       "      <td>26.0000</td>\n",
       "      <td>32.8271</td>\n",
       "      <td>35.50000</td>\n",
       "      <td>50.24790</td>\n",
       "      <td>81.8583</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>B</th>\n",
       "      <td>47.0</td>\n",
       "      <td>113.505764</td>\n",
       "      <td>109.301500</td>\n",
       "      <td>0.0000</td>\n",
       "      <td>57.0000</td>\n",
       "      <td>80.00000</td>\n",
       "      <td>120.00000</td>\n",
       "      <td>512.3292</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>C</th>\n",
       "      <td>59.0</td>\n",
       "      <td>100.151341</td>\n",
       "      <td>70.225588</td>\n",
       "      <td>26.5500</td>\n",
       "      <td>42.5021</td>\n",
       "      <td>83.47500</td>\n",
       "      <td>143.59165</td>\n",
       "      <td>263.0000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>D</th>\n",
       "      <td>33.0</td>\n",
       "      <td>57.244576</td>\n",
       "      <td>29.592832</td>\n",
       "      <td>12.8750</td>\n",
       "      <td>30.0000</td>\n",
       "      <td>53.10000</td>\n",
       "      <td>77.28750</td>\n",
       "      <td>113.2750</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>E</th>\n",
       "      <td>32.0</td>\n",
       "      <td>46.026694</td>\n",
       "      <td>32.608315</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>26.1125</td>\n",
       "      <td>45.18125</td>\n",
       "      <td>56.15730</td>\n",
       "      <td>134.5000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>F</th>\n",
       "      <td>340.0</td>\n",
       "      <td>8.289938</td>\n",
       "      <td>3.098676</td>\n",
       "      <td>4.0125</td>\n",
       "      <td>7.7333</td>\n",
       "      <td>7.85420</td>\n",
       "      <td>8.05000</td>\n",
       "      <td>39.0000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>G</th>\n",
       "      <td>4.0</td>\n",
       "      <td>13.581250</td>\n",
       "      <td>3.601222</td>\n",
       "      <td>10.4625</td>\n",
       "      <td>10.4625</td>\n",
       "      <td>13.58125</td>\n",
       "      <td>16.70000</td>\n",
       "      <td>16.7000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "      count        mean         std      min      25%       50%        75%  \\\n",
       "deck                                                                         \n",
       "A      15.0   41.990553   14.358954  26.0000  32.8271  35.50000   50.24790   \n",
       "B      47.0  113.505764  109.301500   0.0000  57.0000  80.00000  120.00000   \n",
       "C      59.0  100.151341   70.225588  26.5500  42.5021  83.47500  143.59165   \n",
       "D      33.0   57.244576   29.592832  12.8750  30.0000  53.10000   77.28750   \n",
       "E      32.0   46.026694   32.608315   8.0500  26.1125  45.18125   56.15730   \n",
       "F     340.0    8.289938    3.098676   4.0125   7.7333   7.85420    8.05000   \n",
       "G       4.0   13.581250    3.601222  10.4625  10.4625  13.58125   16.70000   \n",
       "\n",
       "           max  \n",
       "deck            \n",
       "A      81.8583  \n",
       "B     512.3292  \n",
       "C     263.0000  \n",
       "D     113.2750  \n",
       "E     134.5000  \n",
       "F      39.0000  \n",
       "G      16.7000  "
      ]
     },
     "execution_count": 98,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.groupby(\"deck\")[\"fare\"].describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 99,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "80.0"
      ]
     },
     "execution_count": 99,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.groupby(\"deck\")[\"fare\"].median()[\"B\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 100,
   "metadata": {},
   "outputs": [],
   "source": [
    "df.loc[df[(df[\"deck\"]==\"B\") & (df[\"fare\"]==0)].index, \"fare\"] = df.groupby(\"deck\")[\"fare\"].median()[\"B\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 101,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>survived</th>\n",
       "      <th>pclass</th>\n",
       "      <th>sex</th>\n",
       "      <th>age</th>\n",
       "      <th>sibsp</th>\n",
       "      <th>parch</th>\n",
       "      <th>fare</th>\n",
       "      <th>embarked</th>\n",
       "      <th>class</th>\n",
       "      <th>who</th>\n",
       "      <th>adult_male</th>\n",
       "      <th>deck</th>\n",
       "      <th>embark_town</th>\n",
       "      <th>alive</th>\n",
       "      <th>alone</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "Empty DataFrame\n",
       "Columns: [survived, pclass, sex, age, sibsp, parch, fare, embarked, class, who, adult_male, deck, embark_town, alive, alone]\n",
       "Index: []"
      ]
     },
     "execution_count": 101,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df[df[\"fare\"]==0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 102,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "survived         0\n",
       "pclass           0\n",
       "sex              0\n",
       "age            177\n",
       "sibsp            0\n",
       "parch            0\n",
       "fare             0\n",
       "embarked         2\n",
       "class            0\n",
       "who              0\n",
       "adult_male       0\n",
       "deck           361\n",
       "embark_town      2\n",
       "alive            0\n",
       "alone            0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 102,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 103,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "NaN    361\n",
       "F      340\n",
       "C       59\n",
       "B       47\n",
       "D       33\n",
       "E       32\n",
       "A       15\n",
       "G        4\n",
       "Name: deck, dtype: int64"
      ]
     },
     "execution_count": 103,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df[\"deck\"].value_counts(dropna=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 104,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>count</th>\n",
       "      <th>mean</th>\n",
       "      <th>std</th>\n",
       "      <th>min</th>\n",
       "      <th>25%</th>\n",
       "      <th>50%</th>\n",
       "      <th>75%</th>\n",
       "      <th>max</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>deck</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>A</th>\n",
       "      <td>15.0</td>\n",
       "      <td>41.990553</td>\n",
       "      <td>14.358954</td>\n",
       "      <td>26.0000</td>\n",
       "      <td>32.8271</td>\n",
       "      <td>35.50000</td>\n",
       "      <td>50.24790</td>\n",
       "      <td>81.8583</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>B</th>\n",
       "      <td>47.0</td>\n",
       "      <td>116.910019</td>\n",
       "      <td>106.881395</td>\n",
       "      <td>5.0000</td>\n",
       "      <td>57.9792</td>\n",
       "      <td>80.00000</td>\n",
       "      <td>120.00000</td>\n",
       "      <td>512.3292</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>C</th>\n",
       "      <td>59.0</td>\n",
       "      <td>100.151341</td>\n",
       "      <td>70.225588</td>\n",
       "      <td>26.5500</td>\n",
       "      <td>42.5021</td>\n",
       "      <td>83.47500</td>\n",
       "      <td>143.59165</td>\n",
       "      <td>263.0000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>D</th>\n",
       "      <td>33.0</td>\n",
       "      <td>57.244576</td>\n",
       "      <td>29.592832</td>\n",
       "      <td>12.8750</td>\n",
       "      <td>30.0000</td>\n",
       "      <td>53.10000</td>\n",
       "      <td>77.28750</td>\n",
       "      <td>113.2750</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>E</th>\n",
       "      <td>32.0</td>\n",
       "      <td>46.026694</td>\n",
       "      <td>32.608315</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>26.1125</td>\n",
       "      <td>45.18125</td>\n",
       "      <td>56.15730</td>\n",
       "      <td>134.5000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>F</th>\n",
       "      <td>340.0</td>\n",
       "      <td>8.289938</td>\n",
       "      <td>3.098676</td>\n",
       "      <td>4.0125</td>\n",
       "      <td>7.7333</td>\n",
       "      <td>7.85420</td>\n",
       "      <td>8.05000</td>\n",
       "      <td>39.0000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>G</th>\n",
       "      <td>4.0</td>\n",
       "      <td>13.581250</td>\n",
       "      <td>3.601222</td>\n",
       "      <td>10.4625</td>\n",
       "      <td>10.4625</td>\n",
       "      <td>13.58125</td>\n",
       "      <td>16.70000</td>\n",
       "      <td>16.7000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "      count        mean         std      min      25%       50%        75%  \\\n",
       "deck                                                                         \n",
       "A      15.0   41.990553   14.358954  26.0000  32.8271  35.50000   50.24790   \n",
       "B      47.0  116.910019  106.881395   5.0000  57.9792  80.00000  120.00000   \n",
       "C      59.0  100.151341   70.225588  26.5500  42.5021  83.47500  143.59165   \n",
       "D      33.0   57.244576   29.592832  12.8750  30.0000  53.10000   77.28750   \n",
       "E      32.0   46.026694   32.608315   8.0500  26.1125  45.18125   56.15730   \n",
       "F     340.0    8.289938    3.098676   4.0125   7.7333   7.85420    8.05000   \n",
       "G       4.0   13.581250    3.601222  10.4625  10.4625  13.58125   16.70000   \n",
       "\n",
       "           max  \n",
       "deck            \n",
       "A      81.8583  \n",
       "B     512.3292  \n",
       "C     263.0000  \n",
       "D     113.2750  \n",
       "E     134.5000  \n",
       "F      39.0000  \n",
       "G      16.7000  "
      ]
     },
     "execution_count": 104,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.groupby(\"deck\")[\"fare\"].describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 105,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "NaN    361\n",
       "F      340\n",
       "C       59\n",
       "B       47\n",
       "D       33\n",
       "E       32\n",
       "A       15\n",
       "G        4\n",
       "Name: deck, dtype: int64"
      ]
     },
     "execution_count": 105,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df[\"deck\"].value_counts(dropna=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 106,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Let's share the remaining nan values proportionally to the other \"deck classes\" except \"F\"\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 107,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[340, 59, 47, 33, 32, 15, 4]"
      ]
     },
     "execution_count": 107,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "value_counts = df[\"deck\"].value_counts().to_list()\n",
    "value_counts"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 108,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[59, 47, 33, 32, 15, 4]"
      ]
     },
     "execution_count": 108,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "value_counts.remove(max(value_counts))\n",
    "value_counts"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 109,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "190"
      ]
     },
     "execution_count": 109,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "total = sum(value_counts)\n",
    "total"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 110,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[0.3105263157894737,\n",
       " 0.24736842105263157,\n",
       " 0.1736842105263158,\n",
       " 0.16842105263157894,\n",
       " 0.07894736842105263,\n",
       " 0.021052631578947368]"
      ]
     },
     "execution_count": 110,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "class_weight = []\n",
    "for i in value_counts:\n",
    "    class_weight.append(i / total)\n",
    "class_weight"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 111,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[112, 89, 63, 61, 28, 8]"
      ]
     },
     "execution_count": 111,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "no_of_fill = []\n",
    "for i in class_weight:\n",
    "    no_of_fill.append(round(i * 361))\n",
    "no_of_fill"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 112,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "361"
      ]
     },
     "execution_count": 112,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "sum(no_of_fill)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 113,
   "metadata": {},
   "outputs": [],
   "source": [
    "deck_names = df[\"deck\"].value_counts().to_dict()\n",
    "deck_names = list(deck_names.keys())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 114,
   "metadata": {},
   "outputs": [],
   "source": [
    "deck_names.remove(deck_names[0])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 115,
   "metadata": {},
   "outputs": [],
   "source": [
    "df[\"deck\"] = df[\"deck\"].astype(\"object\")\n",
    "for i, j in zip(deck_names, no_of_fill):\n",
    "    df[\"deck\"].fillna(i, limit=j, inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 116,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "F    340\n",
       "C    171\n",
       "B    136\n",
       "D     96\n",
       "E     93\n",
       "A     43\n",
       "G     12\n",
       "Name: deck, dtype: int64"
      ]
     },
     "execution_count": 116,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df[\"deck\"].value_counts(dropna=False)"
   ]
  },
  {
   "attachments": {
    "image.png": {
     "image/png": "iVBORw0KGgoAAAANSUhEUgAAAbQAAADICAYAAACAoxRIAAAYPUlEQVR4Ae2dy5Elt3ZF2wkZ8GSBPNBAjskNjjSXCZIRjHg9eROa0BNOOCzFaWlT+x4eJJC/m8i8qyIyAJwfgIW82F3VXeS3L74gAAEIQAACDyDw7QF7uGwLP378+Pr+/fvX3//+969ff/2VBwa8A7wDvAMXvgMI2g45DDH7/fffv/74449Ln99+++3S+a/eP/Nf+/7BH/6zvAMI2g5Bi+/MZjhIBI0LZYb3kDXwHl79DiBoOwQtfsx49QHG/AgaF8kM7yFr4D28+h1A0BC0KUT56g8C83MZ8w7c/x1A0BA0BO3ivwPlIr3/RcoZznGGCBqChqAhaLwDvAOPeAcQNATtES8yf0Ke40/InAPncOU7gKAhaAgafzrnHeAdeMQ7gKAhaI94ka/8UyFz810J78Ac78DbBe1f/vU/vvTs0JJNqZo32iO+7vrP9r99+1aKWNj15A+o7Dn3v/72t694It77Od/Hinfb1n6vlq/J+z5fa28ec1b/yrnP2tOVdb/9d/1u+5ryO+y+T+t/+/dvX/4cvf+R8zhyzj8FzS979Y+49HONETFZiumtreeP9SzVz+tdGp8laL/88kspOK2DX/t7aNUHurL5fC2/i4T3PTf3Iy7bto57tXxN3q/ma+2xij3aduXcR+9lbb2j9j56eR4139p9nhHvYqT+lnn25PbmGz2XXp0R/5+Cpov7qMte9XLbq98TpL3+WE9vDXnNrfEZghZihqCN//gCQRtnNXIhXBFzhMCsuTSPmO8KTtWcRwnRUXWqNYZtzfm0aozYhwVNItATlNblL7vqaOytfGrdF33Z1a71K76VrzmW/KoR7dGCJiFTO3KAETPyHVp8iP3JtXsf8p4/18tjfXek1v2yqXVf9GVX6/6wLY3d1+tv3WPOy+OYN2yVXWtq+bI9j712+Cq/5mi1ylOb42RX6/6wjYxbubJ76/Wi777oZ//PmM6PGls1VE/+svbCZyfna6w6Gi/V/7n+rWf3fz8y1HytVoIVbRUj/5JvT8yUguYXvff9ku/1R/KqmLDJrtbn6vlzrI+973XcXvWPFjS9TEcLmj5Uqu9jfdC8VZy3nuP2kf6S6GSRyuOon/N9Tvd532NG+1v3mPOWxtmntY3ac1yM3ZbHqr/Uen6Oy/WqsefkWr34yM05S/Xcp37vssz1fdxbX+Xv5ff8WrfaHC/7SLskMsp3EWvFr7WrdrQ5N48V2zsnxe1pV3+Hpou9EhX5ltqRvCrGbd7XXG7zvvze9vweu9S/g6D5h0UvyqhN8dFWOe5v9SuRcZv3VcNt3pffW/nVum9tf+seI89zvZ/X0PKN2nNcHsd8lS2vQ+NebOV3m/eruXv+Kkdr6/kUt3RR5vlzzezvjXv5PX+urz2ozf7//Oevr+r5Mz79o44QE/mqtik2je/0FN+qK7/PVdnCv3ROnr+nfwtBywK0dpxFKedn/+gYQev//U0lNG6LfvXopfZY2bxVrtu29vNlsqaOctV6btj8cZ/6VV74sr0ae231VbfX5no5vvK7zfuj6x2Zw2NiDj1uV3/poszrixy3eV/1vK38bvO+8tzm/Ty3x0ecHtlH2pZ4eK5ivHV/9OXLdvdVMbJVba61dE45duv4NoIWIlQ9IT6VXbZKnMJ3xBeCdoygLb28I4IW+b24pTnky5eP7COtctUqpzduxbXso/WU32tzvRxf+d3m/chdO65y8hp8nOv/zF/4+7My3v4ervJ35+vke03v/1yr5Y6Mq+/OwqY1Skg0zm3257HiW3b51ea4PFZc1SJoDdXpCdIe/5IQ5uXcRdD8QxV9H+vFq2zyRdvze6z383dQvbHnRj/HV37Z9ora0h7D1/PHOnKMj5dqeJz2k+tV+ZXN83v9Xn72V2PNkX15/dW4ZVPN3MYcf7F1BM1z8hrdl+vGuIr3HO8r3uus8ee5vE6r3xOU7M9j1W3Z5Veb4zSOVjGt9q2CpovcW7/As0jksccu9ZfyfG71q1pLNSJ+j39p3ryWowUt/jFIflovh9t7/8pRHxR9uNR6jcq2xu+xuS9RkuCoVZz71ZcvWtnUZt/S2H29/hKD8C35o3bLr1y1vg7ZclvFtObIuTH2/F4/5+f4UX+1vryWPNZcPodsqtfyvcQNiJrmVqv6Xqfq+/yeq3z35/wqvopZqpHjfSxBcVvuK8ZbxbjN+0t++bz1XPVf/Avn43F7+3/5kWO+tI8e98Tm6Pmqeket4WhB23qYPUHbWtfz8gfTfU/pf8Ien3JWeR/v+NN/njPGvDP9v3b4yQlBq6ToGBuCNvYS+gf4Ez64n7BHP9On9a8QNd6Z/l3yznO55Du0NT/WO0bC/reK5kXQ+i9hvuzig6sn++4+1r6edjn5vqr+3c+tWv87L8+Y/2nvTMV0j+3t53GkYHxarU/6keOel5rc9X+AgBnMeAfWvwNv/w7tSaKHoK1/4fiQwox3gHfgrHcAQduhsAgaH8yzPpjU5d3iHVj/DiBoCNqqf+LNh2z9hwxmMOMdeM87gKAhaAjaH+/5sHGpwZl34Nx3AEHbIWjfv3+fQgze8XtofBDP/SDCF768A/vfAQRth6D9+PHjK0Qt/i6NBwa8A7wDvAPXvgMI2g5BIxUCEIAABOYhgKDNcxasBAIQgAAEdhBA0HbAIxUCEIAABOYhgKDNcxasBAIQgAAEdhBA0HbAIxUCEIAABOYhgKDNcxasBAIQgAAEdhBA0HbAIxUCEIAABOYhgKDNcxasBAIQgAAEdhBA0HbAIxUCEIAABOYh8HZB+8e//dOXnndj0LzR8gUBCEAAAs8i8CJoey/8EaHYG6M1to6h54+8kTW06mOHAAQgAIE5CfwpaPmSz+OR5Y/k9GLCvxSz1x/7WKo/ss9ejP539x4XttGvvfmj82yN0/rUturIH23ra8nXysEOAQhAoCLw86ZpXfBu934U8nH0q6ea0POyXz61R/tVr1U//OFb8qvGUquL3GPWXNx7833ed/SrvVW2vJZqnzmGMQQgAIFRAocImiYbEYKtMS40VY2eX2uMtsqX3+vItrbVZa428r2veq0LXbFqW/mqc3Xr6xxdq3LUXr0H5ocABO5P4FaCJtyVILnN+8rxtuf32C19XdJqo4b387jlc7v3t6zpzJy8No2j1dOaX7EtP3YIQAACowRuIWhZgNaOM4ycn/17x7qk1UY97+f62aex2l5+rveOcaxNT56vsvtePL5l9xj6EIAABEYI3EbQQoSqJzZZ2WWrIITvzC+/pNVXq3lj7I/s0Xqs+mo9bpZ+Xlse5z35uqtY99OHAAQgMErgp6BFcL7kl8Ytscg51SKOiOnV2OOP3F5+tS+3+SWtvtqI835vrFi1Ps8s/by2PK72qLVXsfLRQgACEFhD4E9BiyRd5q0LXX7FVhMpZqlGlaeanr+lhuq05uj5Nf9Sfs+XL+kYuy33fRy1q3G29dbwTn9eW4zdlse+No9zO30IQAACawm8CNra5C3xLZHaUmtrztlryJd0daHLptb3MpLv8e/ua81qq/nlU5tjZPc2xzCGAAQgsIYAgraGFrEQgAAEIDAtgUsE7Ygf620hqnnP/g5ty9rIgQAEIACBfQTeLmj7lks2BCAAAQhAoCaAoNVcsEIAAhCAwM0IIGg3OzCWCwEIQAACNQEEreaCFQIQgAAEbkYAQbvZgbFcCEAAAhCoCSBoNResEIAABCBwMwII2gkH5r8srP4J01xWUntS21rIkl8+ta0a2CEAAQiMEkDQRkmtiItL+pO+WvtdEivP8f4ncWOvEIDAsQQ+6+Y9ll2z2qdd0NV+ZVPrsEZtnkMfAhCAQI8AgtYjtMFfXdgbytwmJe83xrKp9c1km8d7HH0IQAACawggaGtoDcbqgvZ2MPU2YUt7C5++vJ9tqhH2Kk7xtBCAAARGCPz/zTMSTcwQgU+7nH2/3g9YeSxbtufxEGiCIAABCBgBBM1gHNX9tMvZ9xv91iO+Hr9kk48WAhCAwAgBBG2E0sqY6sJeWeJW4Uv7bfnc7v1bbZzFQgACUxFA0E44jqdf0LE/f5YQtlh4fitmqS4+CEAAApkAgpaJMIYABCAAgVsSQNBueWwsGgIQgAAEMgEELRNhDAEIQAACtySAoN3y2Fg0BCAAAQhkAghaJsIYAhCAAARuSQBBu+WxsWgIQAACEMgEELRMhDEEIAABCNySAIJ2y2Nj0RCAAAQgkAkgaJnIgWP/5eEDy05fyvetfl607GqznzEEIACBtQQQtLXEBuPjovavPHbf0/q9vWYRy+On8WA/EIDAewi83rrvmfPxs/Qu9KcD6O2/8le2p3NifxCAwLEEELRjef6s9umXc2//lb+ynXA0lIQABB5MAEE74XA//XKO/fuTEWc+is1xjCEAAQisIYCgraE1GJsv7MG0x4ZVPCRi8ql9LAQ2BgEInE4AQTsBMZfzK9QRHiMxr1UZQQACEHglgKC98jhslC/oPD5soskLxb57e+/5J98iy4MABCYhgKCdeBC6zD/twvZ9t/buMSceAaUhAIEPIoCgfdBhs1UIQAACTyaAoD35dNkbBCAAgQ8igKB90GGzVQhAAAJPJoCgPfl02RsEIACBDyKAoH3QYbNVCEAAAk8mgKA9+XTZGwQgAIEPIoCgfdBhs1UIQAACTyaAoD35dNkbBCAAgQ8igKCdcNj+S8PqnzDN9CVbe5c9t9NviAVCAAJTE0DQTjieuKjzV2XLMU8bS7BG9vWJfEa4EAMBCIwT+OvNO55LZINAdTlXtkb6I8zar9qlTY3ELOXjgwAEIBAEELQT3oN8Qcc4206YdpqSvt+RfY/ETLM5FgIBCExLAEE74Wh0oXt7wjTTlnSB8n614J6/ysEGAQhAoCKAoFVUdtqqS7qy7ZxmyvS8zzzOi+75czxjCEAAAi0CCFqLzA57dUlXth1TTJsa+2w9edGfwiTvmzEEIHAOAQTtBK7VRV3ZTph6upJL+17yTbcRFgQBCExPAEE74Yjios7PCdPcomRLtMTnFptgkRCAwC0IIGi3OCYWCQEIQAACPQIIWo8QfghAAAIQuAUBBO0Wx8QiIQABCECgRwBB6xHCDwEIQAACtyCAoN3imFgkBCAAAQj0CCBoPUL4IQABCEDgFgQQtFscE4uEAAQgAIEeAQStRwg/BCAAAQjcggCCdsIx6ZeG1Z4wxdQltW+1ebGyq81+xhCAAAS2EEDQtlDr5MRF7V957L5P6Pf23/N/AiP2CAEI7CfwevPur0eF+J/MJUELKJXtU2D19t7zfwon9gkBCOwjgKDt41dmVxd0ZSuTH2js7b3nfyAStgQBCJxAAEE7Ayrfob38x5krxCFieio/NghAAAJrCSBoa4kNxFffcVS2gVKPCOntved/BAQ2AQEInE4AQTsBcXVBV7YTpp6yZG/vPf+Um2JREIDAdAQQtBOOJF/QeXzClFOX7O2/5596cywOAhCYhgCCdsJRxAXtzwlTTF3S916JVc8/9eZYHAQgMC0BBG3ao2FhEIAABCCwhgCCtoYWsRCAAAQgMC0BBG3ao2FhEIAABCCwhgCCtoYWsRCAAAQgMC0BBG3ao2FhEIAABCCwhgCCtoYWsRCAAAQgMC0BBG3ao2FhEIAABCCwhgCCtoYWsRCAAAQgMC0BBO2Eo8m/OFz9cvEJ005TMu+/WthITJWHDQIQgECLAILWIrPD/mkC1kOVeeRx5Fe2Xl38EIAABJwAguY0DupzOb+CHOExEvNalREEIACBVwII2iuPQ0Zczq8YR3iMxLxWZQQBCEDglQCC9srjkFFczvk5pPCNivj+e8tGzHqE8EMAAiMEELQRSitjuKBfgS3xkPC9ZjCCAAQgsJ4AgraeWTdj6QLvJj8woMUDMXvgYbMlCFxIAEE7AX7rAj9hqluUrHggZrc4OhYJgVsRQNBOOC5d1t6eMM20JX3f0c9f2a9xjmMMAQhAYA2Bv942a7KJhQAEIAABCExCAEGb5CBYBgQgAAEI7COAoO3jRzYEIAABCExCAEGb5CBYBgQgAAEI7COAoO3jRzYEIAABCExCAEGb5CBYBgQgAAEI7COAoO3jRzYEIAABCExCAEGb5CBYBgQgAAEI7COAoO3jt5jNLwzX/58zcfF2ESROCEAAAgMEELQBSFtC4rKOL7Vbatw9R4KV9/HJTDILxhCAwHEEELTjWL5U0qWt9sX5AQPtW61vubK5nz4EIACBLQQQtC3UOjlxYevS9n4n7ZFucfDNVTb304cABCCwhQCCtoVaJydf2HncSX+Uu9p72Px51IbZDAQgcBkBBO0E9PkSz+MTppy25MjeR2Km3SALgwAEpiGAoB18FHE5t56Dp7pFuRGxGom5xWZZJAQgcCkBBO1g/K3LuWU/ePrpyvX2Hf5ezHSbYkEQgMCUBBC0g4+ldTm37AdPP005CZW3WpzbPo2LGNBCAALHE0DQjmdKRQhAAAIQuIAAgnYBdKaEAAQgAIHjCSBoxzOlIgQgAAEIXEAAQbsAOlNCAAIQgMDxBBC045lSEQIQgAAELiCAoF0AnSkhAAEIQOB4Agja8UypCAEIQAACFxBA0C6AzpQQgAAEIHA8AQTteKY/K/ovD580xe3LitHtN8IGIACBKQggaCccQ/6vX+TxCVPesiSCdstjY9EQmJYAgnbw0SBeY0DFSe1YFlEQgAAE2gQQtDabTR4u6D62YCROavtZREAAAhBYJoCgLfNZ7fULWhe321YXfGCC8/D+A7fKliAAgTcSQNAOhl1d0JXt4GlvUy6zyOPbbISFQgAC0xFA0A4+kuqCrmwHT3ubcsGi9dxmEywUAhCYkgCCdsKxZAHL4xOmvG1J2Nz26Fg4BKYjgKCddCT5u5CTprl9WQTt9kfIBiAwDQEEbZqjYCEQgAAEILCHAIK2hx65EIAABCAwDQEEbZqjYCEQgAAEILCHAIK2hx65EIAABCAwDQEEbZqjYCEQgAAEILCHAIK2hx65EIAABCAwDQEEbZqjYCEQgAAEILCHAIK2h94H5R75+2JH1ho9givmHF3blrjYz9P2tIUDORBwAgia0zioX102d798jlz/kbVGjyzPmcejdWaJi/Xv2cNIruZoxco/CxPWAQEE7YR3oPqgty6FE6Y/peSR6z+y1uhm85x5PFrnKXG9/ff8wSFiRuKewox9zE8AQTvhjPQhVxtTeF9Thi3bZata5amex7hPffk1XtsqX63ny6bWfdGXXa37w+Zfeey+PX3NrTZqqZ/bPE/4/cvH6nsNj42++xTfisn23thrV7GaT3EeI1tuc4yPq77PUfmxQeAKAq+f2itW8MA5qw+7bNquj3Nf42i977myh83jFLNk95hW3+urlmLzfHmc45Wn1mt7X/4j2ly3N85zLsWHz/1rx5or58k+2voaPCfXreIqm2rIpzoau182tfLRQuBKAgjaCfT1IVcbU3g/T+m+tX3V8jzZtrZVLbd5X3O4zfvyeyu/Wvcd0a/qZlse53nDrxjvR5zsnuM27yumssm3tW3VzPY8jvkqm9YRvuz3cauvfFoIXEUAQTuBfPWBd1tMGWN/tAyPW+p7rvqqsbf1eVXLbZovt1WsbN4qz21H9n2tqptteaw4bxWjVr48Drvbol89yj+q9Tm9ZrbncV6v57Z8qqFWOXksOy0EriCAoJ1A3T/k6quN6byfx+4b6Z+w/L+sb2mN1fy+7iV/L67KHbFVdbMtj6u6ilGrmDwOu9u8r5wz2tY82Z7Heb15bUvx4Ws9uQ5jCLybAIJ2AvF8IegC0FTu7/lGchST21w7+1vjnNcb5zo5vvLLFrGtr16d0byqztK8qqsYtdm+NM45ivU2YkbiPMf7rdxsz+OoUdlUO3zuz2PF9ep4HH0IvINA+zZ5x+wPncMvg9hidSHIplYoPLfV95rK91ivVdnlX2pz3VzH/ep7PdnUZt/SWL4qV75eq9xo40ut5+UY96nfy6v8ms/rV3Hya67RVnm5VX6eK489TjVkUyu7Wtlz26qd4xhD4B0EELR3UGaOzQSuvjCr+Svb5g2SCAEIHEYAQTsMJYWOJBCicbVwtOZv2Y/cP7UgAIH1BBC09czIeDgBiWlLuFr2h2NhexCYngCCNv0RsUAIQAACEBghgKCNUCIGAhCAAASmJ4CgTX9ELBACEIAABEYIIGgjlIiBAAQgAIHpCSBo0x8RC4QABCAAgRECCNoIJWIgAAEIQGB6Agja9EfEAiEAAQhAYIQAgjZCiRgIQAACEJieAII2/RGxQAhAAAIQGCGAoI1QIgYCEIAABKYn8D+JkFJwB/IhBwAAAABJRU5ErkJggg=="
    }
   },
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "![image.png](attachment:image.png)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 117,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "survived         0\n",
       "pclass           0\n",
       "sex              0\n",
       "age            177\n",
       "sibsp            0\n",
       "parch            0\n",
       "fare             0\n",
       "embarked         2\n",
       "class            0\n",
       "who              0\n",
       "adult_male       0\n",
       "deck             0\n",
       "embark_town      2\n",
       "alive            0\n",
       "alone            0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 117,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 118,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>embarked</th>\n",
       "      <th>embark_town</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>S</td>\n",
       "      <td>Southampton</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>C</td>\n",
       "      <td>Cherbourg</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>S</td>\n",
       "      <td>Southampton</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>S</td>\n",
       "      <td>Southampton</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>S</td>\n",
       "      <td>Southampton</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "  embarked  embark_town\n",
       "0        S  Southampton\n",
       "1        C    Cherbourg\n",
       "2        S  Southampton\n",
       "3        S  Southampton\n",
       "4        S  Southampton"
      ]
     },
     "execution_count": 118,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df[[\"embarked\", \"embark_town\"]].head()\n",
    "# these two column have same content.Then we can drop one of them."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 119,
   "metadata": {},
   "outputs": [],
   "source": [
    "df.drop(\"embarked\", axis=1, inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 120,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "survived         0\n",
       "pclass           0\n",
       "sex              0\n",
       "age            177\n",
       "sibsp            0\n",
       "parch            0\n",
       "fare             0\n",
       "class            0\n",
       "who              0\n",
       "adult_male       0\n",
       "deck             0\n",
       "embark_town      2\n",
       "alive            0\n",
       "alone            0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 120,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 121,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Southampton    644\n",
       "Cherbourg      168\n",
       "Queenstown      77\n",
       "NaN              2\n",
       "Name: embark_town, dtype: int64"
      ]
     },
     "execution_count": 121,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df[\"embark_town\"].value_counts(dropna=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 122,
   "metadata": {},
   "outputs": [],
   "source": [
    "#let's fill nan values with most freqeunt\n",
    "df[\"embark_town\"].fillna(\"Southampton\", inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 123,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Southampton    646\n",
       "Cherbourg      168\n",
       "Queenstown      77\n",
       "Name: embark_town, dtype: int64"
      ]
     },
     "execution_count": 123,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df[\"embark_town\"].value_counts(dropna=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 124,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "count    714.000000\n",
       "mean      29.699118\n",
       "std       14.526497\n",
       "min        0.420000\n",
       "25%       20.125000\n",
       "50%       28.000000\n",
       "75%       38.000000\n",
       "max       80.000000\n",
       "Name: age, dtype: float64"
      ]
     },
     "execution_count": 124,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df[\"age\"].describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 125,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "803    0.42\n",
       "755    0.67\n",
       "644    0.75\n",
       "469    0.75\n",
       "78     0.83\n",
       "Name: age, dtype: float64"
      ]
     },
     "execution_count": 125,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df[\"age\"].sort_values()[:5]\n",
    "# there are a few floating point ages. We will handle with them. Firstly we will fill the an values with average age"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 126,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "29.69911764705882"
      ]
     },
     "execution_count": 126,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df[\"age\"].mean()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 127,
   "metadata": {},
   "outputs": [],
   "source": [
    "df[\"age\"].fillna(df[\"age\"].mean(), inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 128,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "survived       0\n",
       "pclass         0\n",
       "sex            0\n",
       "age            0\n",
       "sibsp          0\n",
       "parch          0\n",
       "fare           0\n",
       "class          0\n",
       "who            0\n",
       "adult_male     0\n",
       "deck           0\n",
       "embark_town    0\n",
       "alive          0\n",
       "alone          0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 128,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 129,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "164    1\n",
       "644    1\n",
       "755    1\n",
       "469    1\n",
       "788    1\n",
       "Name: age, dtype: int64"
      ]
     },
     "execution_count": 129,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#now we can round the ages\n",
    "df[\"age\"] = df[\"age\"].apply(lambda x: int(np.ceil(x))) # np.ceil(x) is used for rounding up the number\n",
    "df[\"age\"].sort_values()[:5]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 130,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>survived</th>\n",
       "      <th>pclass</th>\n",
       "      <th>age</th>\n",
       "      <th>sibsp</th>\n",
       "      <th>parch</th>\n",
       "      <th>fare</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>891.000000</td>\n",
       "      <td>891.000000</td>\n",
       "      <td>891.000000</td>\n",
       "      <td>891.000000</td>\n",
       "      <td>891.000000</td>\n",
       "      <td>891.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>0.383838</td>\n",
       "      <td>2.308642</td>\n",
       "      <td>29.771044</td>\n",
       "      <td>0.523008</td>\n",
       "      <td>0.381594</td>\n",
       "      <td>32.529405</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>0.486592</td>\n",
       "      <td>0.836071</td>\n",
       "      <td>13.002476</td>\n",
       "      <td>1.102743</td>\n",
       "      <td>0.806057</td>\n",
       "      <td>49.648684</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>4.012500</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>0.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>22.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>7.925000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>0.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>30.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>14.458300</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>1.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>35.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>31.275000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>1.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>80.000000</td>\n",
       "      <td>8.000000</td>\n",
       "      <td>6.000000</td>\n",
       "      <td>512.329200</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "         survived      pclass         age       sibsp       parch        fare\n",
       "count  891.000000  891.000000  891.000000  891.000000  891.000000  891.000000\n",
       "mean     0.383838    2.308642   29.771044    0.523008    0.381594   32.529405\n",
       "std      0.486592    0.836071   13.002476    1.102743    0.806057   49.648684\n",
       "min      0.000000    1.000000    1.000000    0.000000    0.000000    4.012500\n",
       "25%      0.000000    2.000000   22.000000    0.000000    0.000000    7.925000\n",
       "50%      0.000000    3.000000   30.000000    0.000000    0.000000   14.458300\n",
       "75%      1.000000    3.000000   35.000000    1.000000    0.000000   31.275000\n",
       "max      1.000000    3.000000   80.000000    8.000000    6.000000  512.329200"
      ]
     },
     "execution_count": 130,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 131,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>survived</th>\n",
       "      <th>pclass</th>\n",
       "      <th>sex</th>\n",
       "      <th>age</th>\n",
       "      <th>sibsp</th>\n",
       "      <th>parch</th>\n",
       "      <th>fare</th>\n",
       "      <th>class</th>\n",
       "      <th>who</th>\n",
       "      <th>adult_male</th>\n",
       "      <th>deck</th>\n",
       "      <th>embark_town</th>\n",
       "      <th>alive</th>\n",
       "      <th>alone</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>378</th>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>male</td>\n",
       "      <td>20</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>4.0125</td>\n",
       "      <td>Third</td>\n",
       "      <td>man</td>\n",
       "      <td>True</td>\n",
       "      <td>F</td>\n",
       "      <td>Cherbourg</td>\n",
       "      <td>no</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>872</th>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>male</td>\n",
       "      <td>33</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>5.0000</td>\n",
       "      <td>First</td>\n",
       "      <td>man</td>\n",
       "      <td>True</td>\n",
       "      <td>B</td>\n",
       "      <td>Southampton</td>\n",
       "      <td>no</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>326</th>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>male</td>\n",
       "      <td>61</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>6.2375</td>\n",
       "      <td>Third</td>\n",
       "      <td>man</td>\n",
       "      <td>True</td>\n",
       "      <td>F</td>\n",
       "      <td>Southampton</td>\n",
       "      <td>no</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>843</th>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>male</td>\n",
       "      <td>35</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>6.4375</td>\n",
       "      <td>Third</td>\n",
       "      <td>man</td>\n",
       "      <td>True</td>\n",
       "      <td>F</td>\n",
       "      <td>Cherbourg</td>\n",
       "      <td>no</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>818</th>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>male</td>\n",
       "      <td>43</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>6.4500</td>\n",
       "      <td>Third</td>\n",
       "      <td>man</td>\n",
       "      <td>True</td>\n",
       "      <td>F</td>\n",
       "      <td>Southampton</td>\n",
       "      <td>no</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "     survived  pclass   sex  age  sibsp  parch    fare  class  who  \\\n",
       "378         0       3  male   20      0      0  4.0125  Third  man   \n",
       "872         0       1  male   33      0      0  5.0000  First  man   \n",
       "326         0       3  male   61      0      0  6.2375  Third  man   \n",
       "843         0       3  male   35      0      0  6.4375  Third  man   \n",
       "818         0       3  male   43      0      0  6.4500  Third  man   \n",
       "\n",
       "     adult_male deck  embark_town alive  alone  \n",
       "378        True    F    Cherbourg    no   True  \n",
       "872        True    B  Southampton    no   True  \n",
       "326        True    F  Southampton    no   True  \n",
       "843        True    F    Cherbourg    no   True  \n",
       "818        True    F  Southampton    no   True  "
      ]
     },
     "execution_count": 131,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.sort_values(\"fare\")[:5]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 132,
   "metadata": {},
   "outputs": [],
   "source": [
    "df[\"survived\"] = df[\"survived\"].astype(str) # change type\n",
    "# df['survived'] = df['survived'].apply(str)   ## other method"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 133,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "dtype('O')"
      ]
     },
     "execution_count": 133,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df[\"survived\"].dtype"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 134,
   "metadata": {},
   "outputs": [],
   "source": [
    "df[\"survived\"] = pd.to_numeric(df[\"survived\"])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 135,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "dtype('int64')"
      ]
     },
     "execution_count": 135,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df[\"survived\"].dtype"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 136,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "F    340\n",
       "C    171\n",
       "B    136\n",
       "D     96\n",
       "E     93\n",
       "A     43\n",
       "G     12\n",
       "Name: deck, dtype: int64"
      ]
     },
     "execution_count": 136,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df[\"deck\"].value_counts(dropna=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 137,
   "metadata": {},
   "outputs": [],
   "source": [
    "total = df[\"deck\"].value_counts().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 138,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array(['F', 'C', 'E', 'G', 'D', 'A', 'B'], dtype=object)"
      ]
     },
     "execution_count": 138,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "deck = df[\"deck\"].unique()\n",
    "deck"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 139,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array(['C', 'E', 'G', 'D', 'A', 'B'], dtype=object)"
      ]
     },
     "execution_count": 139,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "deck = deck[1:]\n",
    "deck"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 140,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[0.1919191919191919,\n",
       " 0.10437710437710437,\n",
       " 0.013468013468013467,\n",
       " 0.10774410774410774,\n",
       " 0.04826038159371493,\n",
       " 0.1526374859708193]"
      ]
     },
     "execution_count": 140,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "class_weight = []\n",
    "for i in deck:\n",
    "    class_weight.append((df[\"deck\"]==i).sum() / total)\n",
    "class_weight"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 141,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[132, 72, 9, 74, 33, 105]"
      ]
     },
     "execution_count": 141,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "no_of_fill = []\n",
    "for i in class_weight:\n",
    "    no_of_fill.append(int(round(i* 688)))\n",
    "no_of_fill"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 142,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "425"
      ]
     },
     "execution_count": 142,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "sum(no_of_fill)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 143,
   "metadata": {},
   "outputs": [],
   "source": [
    "#values = {k:v for k,v in zip(deck,no_of_fill)}\n",
    "#values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 144,
   "metadata": {},
   "outputs": [],
   "source": [
    "df[\"deck\"]  = df[\"deck\"].astype(\"object\")\n",
    "\n",
    "\n",
    "for i,j in zip(deck,no_of_fill):\n",
    "    df[\"deck\"].fillna(i, limit=j, inplace=True)\n",
    "\n",
    "#df[\"deck\"].fillna(\"C\", limit=202)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 145,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df[\"deck\"].isnull().sum()"
   ]
  },
  {
   "attachments": {
    "image.png": {
     "image/png": "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"
    }
   },
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "![image.png](attachment:image.png)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 146,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "F    340\n",
       "C    171\n",
       "B    136\n",
       "D     96\n",
       "E     93\n",
       "A     43\n",
       "G     12\n",
       "Name: deck, dtype: int64"
      ]
     },
     "execution_count": 146,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df[\"deck\"].value_counts(dropna=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 147,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[('C', 132), ('E', 72), ('G', 9), ('D', 74), ('A', 33), ('B', 105)]\n"
     ]
    }
   ],
   "source": [
    "print(list(zip(deck,no_of_fill)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 148,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "891"
      ]
     },
     "execution_count": 148,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df[\"deck\"].value_counts().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 149,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "survived       0\n",
       "pclass         0\n",
       "sex            0\n",
       "age            0\n",
       "sibsp          0\n",
       "parch          0\n",
       "fare           0\n",
       "class          0\n",
       "who            0\n",
       "adult_male     0\n",
       "deck           0\n",
       "embark_town    0\n",
       "alive          0\n",
       "alone          0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 149,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 150,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array(['Southampton', 'Cherbourg', 'Queenstown'], dtype=object)"
      ]
     },
     "execution_count": 150,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df[\"embark_town\"].unique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 151,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "3"
      ]
     },
     "execution_count": 151,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df[\"embark_town\"].nunique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 152,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Southampton    646\n",
       "Cherbourg      168\n",
       "Queenstown      77\n",
       "Name: embark_town, dtype: int64"
      ]
     },
     "execution_count": 152,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df[\"embark_town\"].value_counts(dropna=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 153,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>survived</th>\n",
       "      <th>pclass</th>\n",
       "      <th>sex</th>\n",
       "      <th>age</th>\n",
       "      <th>sibsp</th>\n",
       "      <th>parch</th>\n",
       "      <th>fare</th>\n",
       "      <th>class</th>\n",
       "      <th>who</th>\n",
       "      <th>adult_male</th>\n",
       "      <th>deck</th>\n",
       "      <th>embark_town</th>\n",
       "      <th>alive</th>\n",
       "      <th>alone</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "Empty DataFrame\n",
       "Columns: [survived, pclass, sex, age, sibsp, parch, fare, class, who, adult_male, deck, embark_town, alive, alone]\n",
       "Index: []"
      ]
     },
     "execution_count": 153,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df[df[\"embark_town\"].isnull()]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 154,
   "metadata": {},
   "outputs": [],
   "source": [
    "df[\"embark_town\"].fillna(\"Southampton\", inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 155,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0"
      ]
     },
     "execution_count": 155,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df[\"embark_town\"].isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 156,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 891 entries, 0 to 890\n",
      "Data columns (total 14 columns):\n",
      "survived       891 non-null int64\n",
      "pclass         891 non-null int64\n",
      "sex            891 non-null object\n",
      "age            891 non-null int64\n",
      "sibsp          891 non-null int64\n",
      "parch          891 non-null int64\n",
      "fare           891 non-null float64\n",
      "class          891 non-null category\n",
      "who            891 non-null object\n",
      "adult_male     891 non-null bool\n",
      "deck           891 non-null object\n",
      "embark_town    891 non-null object\n",
      "alive          891 non-null object\n",
      "alone          891 non-null bool\n",
      "dtypes: bool(2), category(1), float64(1), int64(5), object(5)\n",
      "memory usage: 79.4+ KB\n"
     ]
    }
   ],
   "source": [
    "df.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 157,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "count    891.000000\n",
       "mean      29.771044\n",
       "std       13.002476\n",
       "min        1.000000\n",
       "25%       22.000000\n",
       "50%       30.000000\n",
       "75%       35.000000\n",
       "max       80.000000\n",
       "Name: age, dtype: float64"
      ]
     },
     "execution_count": 157,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df[\"age\"].describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 158,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "30"
      ]
     },
     "execution_count": 158,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "round(df[\"age\"].mean())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 159,
   "metadata": {},
   "outputs": [],
   "source": [
    "df[\"age\"].fillna(round(df[\"age\"].mean()), inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 160,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0"
      ]
     },
     "execution_count": 160,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df[\"age\"].isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 161,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 891 entries, 0 to 890\n",
      "Data columns (total 14 columns):\n",
      "survived       891 non-null int64\n",
      "pclass         891 non-null int64\n",
      "sex            891 non-null object\n",
      "age            891 non-null int64\n",
      "sibsp          891 non-null int64\n",
      "parch          891 non-null int64\n",
      "fare           891 non-null float64\n",
      "class          891 non-null category\n",
      "who            891 non-null object\n",
      "adult_male     891 non-null bool\n",
      "deck           891 non-null object\n",
      "embark_town    891 non-null object\n",
      "alive          891 non-null object\n",
      "alone          891 non-null bool\n",
      "dtypes: bool(2), category(1), float64(1), int64(5), object(5)\n",
      "memory usage: 79.4+ KB\n"
     ]
    }
   ],
   "source": [
    "df.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 162,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>survived</th>\n",
       "      <th>pclass</th>\n",
       "      <th>age</th>\n",
       "      <th>sibsp</th>\n",
       "      <th>parch</th>\n",
       "      <th>fare</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>891.000000</td>\n",
       "      <td>891.000000</td>\n",
       "      <td>891.000000</td>\n",
       "      <td>891.000000</td>\n",
       "      <td>891.000000</td>\n",
       "      <td>891.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>0.383838</td>\n",
       "      <td>2.308642</td>\n",
       "      <td>29.771044</td>\n",
       "      <td>0.523008</td>\n",
       "      <td>0.381594</td>\n",
       "      <td>32.529405</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>0.486592</td>\n",
       "      <td>0.836071</td>\n",
       "      <td>13.002476</td>\n",
       "      <td>1.102743</td>\n",
       "      <td>0.806057</td>\n",
       "      <td>49.648684</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>4.012500</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>0.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>22.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>7.925000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>0.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>30.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>14.458300</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>1.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>35.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>31.275000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>1.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>80.000000</td>\n",
       "      <td>8.000000</td>\n",
       "      <td>6.000000</td>\n",
       "      <td>512.329200</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "         survived      pclass         age       sibsp       parch        fare\n",
       "count  891.000000  891.000000  891.000000  891.000000  891.000000  891.000000\n",
       "mean     0.383838    2.308642   29.771044    0.523008    0.381594   32.529405\n",
       "std      0.486592    0.836071   13.002476    1.102743    0.806057   49.648684\n",
       "min      0.000000    1.000000    1.000000    0.000000    0.000000    4.012500\n",
       "25%      0.000000    2.000000   22.000000    0.000000    0.000000    7.925000\n",
       "50%      0.000000    3.000000   30.000000    0.000000    0.000000   14.458300\n",
       "75%      1.000000    3.000000   35.000000    1.000000    0.000000   31.275000\n",
       "max      1.000000    3.000000   80.000000    8.000000    6.000000  512.329200"
      ]
     },
     "execution_count": 162,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 163,
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x1500335fc18>"
      ]
     },
     "execution_count": 163,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 864x576 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(12,8))\n",
    "sns.heatmap(df.corr(), annot=True, cmap=\"coolwarm\" )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 164,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "survived          int64\n",
       "pclass            int64\n",
       "sex              object\n",
       "age               int64\n",
       "sibsp             int64\n",
       "parch             int64\n",
       "fare            float64\n",
       "class          category\n",
       "who              object\n",
       "adult_male         bool\n",
       "deck             object\n",
       "embark_town      object\n",
       "alive            object\n",
       "alone              bool\n",
       "dtype: object"
      ]
     },
     "execution_count": 164,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.dtypes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 165,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "survived          int64\n",
       "pclass            int64\n",
       "sex              object\n",
       "age               int64\n",
       "sibsp             int64\n",
       "parch             int64\n",
       "fare            float64\n",
       "class          category\n",
       "who              object\n",
       "adult_male        int32\n",
       "deck             object\n",
       "embark_town      object\n",
       "alive            object\n",
       "alone             int32\n",
       "dtype: object"
      ]
     },
     "execution_count": 165,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df[[\"adult_male\", \"alone\"]] = df[[\"adult_male\", \"alone\"]].astype(int)\n",
    "df.dtypes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 166,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<seaborn.axisgrid.PairGrid at 0x150056bc940>"
      ]
     },
     "execution_count": 166,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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nSuWXxJCXJR5IIT4QwDPV69IRERERERERERERORl07li+Q8BrBFCvDP++HQAvxZC/DsAKaVcUrXSEREREREREREREZHnTCexfEXVS0FEREREREREREREvlF2YllK+bYbBSEiIiIiIiIiIiIif5jOGMtEREREREREREREFGBMLBMRERERERERERFRWZhYJiIiIiIiIiIiIqKyMLFMRERERERERERERGUpe/I+IqJaOePLT5e9zltfu9KFkhARERERERERUS5esUxEREREREREREREZWFimYiIiIiIiIiIiIjKwsQyEREREREREREREZWFiWUiIiIiIiIiIiIiKgsTy0RERERERERERERUFiaWiYiIiIiIiIiIiKgsTCwTERERERERERERUVlqklgWQvyDEOKYEOLVIq9fKoR4Xwjx8vjf39WiXERERERERERERERUPq1G7/MwgAcA7JxkmeeklFfVpjhERERERERERERENF01uWJZSvkzACdq8V5ERERERERERERE5C4vjbF8sRDiFSHEvwghPlzvwhARERERERERERGRM68klvcD+ICUcimA+wH8Y7EFhRC3CCF6hBA9/f39NSsgUaUYu+RXjF3yK8Yu+RVjl/yKsUt+xdglP2Lckhd4IrEspfy9lHJk/N//DCAkhJhbZNntUsouKWVXW1tbTctJVAnGLvkVY5f8irFLfsXYJb9i7JJfMXbJjxi35AWeSCwLIU4RQojxf1+IdLkG6lsqIiIiIiIiIiIiInKi1eJNhBDfA3ApgLlCiD4AdwEIAYCUchuAVQA2CCFMAAkAn5RSylqUjYiIiIiIiIiIiIjKU5PEspTyU1O8/gCAB2pRFiIiIiIiIiIiIiKqjCeGwiAiIiIiIiIiIiIi/2BimYiIiIiIiIiIiIjKwsQyEREREREREREREZWFiWUiIiIiIiIiIiIiKgsTy0RERERERERERERUFiaWiYiIiIiIiIiIiKgsTCwTERERERERERERUVmYWCYiIiIiIiIiIiKisjCxTERERERERERERERlYWKZiIiIiIiIiIiIiMrCxDIRERERERERERERlYWJZSIiIiIiIiIiIiIqCxPLRERERERERERERFQWJpaJiIiIiIiIiIiIqCxMLBMRERERERERERFRWZhYJiIiIiIiIiIiIqKyMLFMRERERERERERERGVhYpmI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hmVLhBSBiK7AtAAICV0ViCftbDusKIBlp9v4sAbEk+m2vEFXYeRsOxJSMJJMt426qiCmA0MJO9vGSwCGaUNTlSnbQsMw0R8/GeOzowreT5x8r7aYDkVR0p/ZSm9zTiTkm+Oi2qbRbnkidv1Ul5E7phEDNYld5sdoKm7Fbk2iSAjxPQCXApgrhOgDcBeAEABIKbcB+Gekk8r/AWAUwM3VeN9IRMN8HiiBxhggqi1FEWhvihQ839ZU2hUoiiIcl50TCwOxios3pdnRqZepx7amUq16rtj+r9X69VRpe1Hp+uGwhvllJJInCoVUzG9pmPb65E9u9XPc2K6f+mR+rsv8xKneCoUw7URH8yRV4OxJXmttzH8cK+zGlETXNczX88ve6LCt05rzOwh+OS6qza/tlp/qMnKHX2PAr+Wm6nErBmoSVVLKT03xugTwF7UoCxERERERERERERFVxitjLBMRERERERERERGRTzCxTERERERERERERERlYWKZiIiIiIiIiIiIiMrCxDIRERERERERERERlUWk583zJyFEP4C3S1h0LoDjLhfHy4L++YHS9sFxKeUVtSgMY7dkQf/8gP9i1w/fGctYuWqVz0uxmxGUfe+WoJTPa7Hr9f2ei2V1R6ll9VLs+mn/ToWfxX1eil3Au/uploK+DzxV786gvm4tBH0fVDV2fZ1YLpUQokdK2VXvctRL0D8/4N994NdyV0vQPz/gv33gh/KyjJXzevkq4fXPxvJVxuvlmy4/fS6W1R1+KmuGH8tcDD9L8HA/cR/49fP7tdzVFPR9UO3Pz6EwiIiIiIiIiIiIiKgsTCwTERERERERERERUVmCkljeXu8C1FnQPz/g333g13JXS9A/P+C/feCH8rKMlfN6+Srh9c/G8lXG6+WbLj99LpbVHX4qa4Yfy1wMP0vwcD9xH/j18/u13NUU9H1Q1c8fiDGWiYiIiIiIiIiIiKh6gnLFMhERERERERERERFVCRPLRERERERERERERFQWJpaJiIiIiIiIiIiIqCxMLBMRERERERERERFRWZhYJiIiIiIiIiIiIqKyMLFMRERERERERERERGVhYpmIiIiIiIiIiIiIysLEMhERERERERERERGVhYllIiIiIiIiIiIiIioLE8tEREREREREREREVBYmlomIiIiIiIiIiIioLEwsExEREREREREREVFZ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\n",
      "text/plain": [
       "<Figure size 1440x1440 with 72 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.pairplot(df)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 167,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x15006ba2550>"
      ]
     },
     "execution_count": 167,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "df.plot(kind=\"box\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 168,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x15008c70320>"
      ]
     },
     "execution_count": 168,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "df.drop(\"fare\", axis=1).plot(kind=\"box\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 169,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x15008d46240>"
      ]
     },
     "execution_count": 169,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.boxplot(df[\"fare\"])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 170,
   "metadata": {},
   "outputs": [],
   "source": [
    "from scipy.stats.mstats import winsorize"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 179,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "70.54999999999995"
      ]
     },
     "execution_count": 179,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df[\"age\"].quantile(0.995)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 180,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1.0"
      ]
     },
     "execution_count": 180,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df[\"age\"].quantile(0.005)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 183,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "masked_array(data=[22, 38, 26, 35, 35, 30, 54,  2, 27, 14,  4, 58, 20, 39,\n",
       "                   14, 55,  2, 30, 31, 30, 35, 34, 15, 28,  8, 38, 30, 19,\n",
       "                   30, 30, 40, 30, 30, 66, 28, 42, 30, 21, 18, 14, 40, 27,\n",
       "                   30,  3, 19, 30, 30, 30, 30, 18,  7, 21, 49, 29, 65, 30,\n",
       "                   21, 29,  5, 11, 22, 38, 45,  4, 30, 30, 29, 19, 17, 26,\n",
       "                   32, 16, 21, 26, 32, 25, 30, 30,  1, 30, 22, 29, 30, 28,\n",
       "                   17, 33, 16, 30, 23, 24, 29, 20, 46, 26, 59, 30, 71, 23,\n",
       "                   34, 34, 28, 30, 21, 33, 37, 28, 21, 30, 38, 30, 47, 15,\n",
       "                   22, 20, 17, 21, 71, 29, 24,  2, 21, 30, 33, 33, 54, 12,\n",
       "                   30, 24, 30, 45, 33, 20, 47, 29, 25, 23, 19, 37, 16, 24,\n",
       "                   30, 22, 24, 19, 18, 19, 27,  9, 37, 42, 51, 22, 56, 41,\n",
       "                   30, 51, 16, 30, 30, 30, 44, 40, 26, 17,  1,  9, 30, 45,\n",
       "                   30, 28, 61,  4,  1, 21, 56, 18, 30, 50, 30, 36, 30, 30,\n",
       "                    9,  1,  4, 30, 30, 45, 40, 36, 32, 19, 19,  3, 44, 58,\n",
       "                   30, 42, 30, 24, 28, 30, 34, 46, 18,  2, 32, 26, 16, 40,\n",
       "                   24, 35, 22, 30, 30, 31, 27, 42, 32, 30, 16, 27, 51, 30,\n",
       "                   38, 22, 19, 21, 18, 30, 35, 29, 59,  5, 24, 30, 44,  8,\n",
       "                   19, 33, 30, 30, 29, 22, 30, 44, 25, 24, 37, 54, 30, 29,\n",
       "                   62, 30, 41, 29, 30, 30, 35, 50, 30,  3, 52, 40, 30, 36,\n",
       "                   16, 25, 58, 35, 30, 25, 41, 37, 30, 63, 45, 30,  7, 35,\n",
       "                   65, 28, 16, 19, 30, 33, 30, 22, 42, 22, 26, 19, 36, 24,\n",
       "                   24, 30, 24,  2, 30, 50, 30, 30, 19, 30, 30,  1, 30, 17,\n",
       "                   30, 30, 24, 18, 26, 28, 43, 26, 24, 54, 31, 40, 22, 27,\n",
       "                   30, 22, 30, 36, 61, 36, 31, 16, 30, 46, 38, 16, 30, 30,\n",
       "                   29, 41, 45, 45,  2, 24, 28, 25, 36, 24, 40, 30,  3, 42,\n",
       "                   23, 30, 15, 25, 30, 28, 22, 38, 30, 30, 40, 29, 45, 35,\n",
       "                   30, 30, 60, 30, 30, 24, 25, 18, 19, 22,  3, 30, 22, 27,\n",
       "                   20, 19, 42,  1, 32, 35, 30, 18,  1, 36, 30, 17, 36, 21,\n",
       "                   28, 23, 24, 22, 31, 46, 23, 28, 39, 26, 21, 28, 20, 34,\n",
       "                   51,  3, 21, 30, 30, 30, 33, 30, 44, 30, 34, 18, 30, 10,\n",
       "                   30, 21, 29, 28, 18, 30, 28, 19, 30, 32, 28, 30, 42, 17,\n",
       "                   50, 14, 21, 24, 64, 31, 45, 20, 25, 28, 30,  4, 13, 34,\n",
       "                    5, 52, 36, 30, 30, 49, 30, 29, 65, 30, 50, 30, 48, 34,\n",
       "                   47, 48, 30, 38, 30, 56, 30,  1, 30, 38, 33, 23, 22, 30,\n",
       "                   34, 29, 22,  2,  9, 30, 50, 63, 25, 30, 35, 58, 30,  9,\n",
       "                   30, 21, 55, 71, 21, 30, 54, 30, 25, 24, 17, 21, 30, 37,\n",
       "                   16, 18, 33, 30, 28, 26, 29, 30, 36, 54, 24, 47, 34, 30,\n",
       "                   36, 32, 30, 22, 30, 44, 30, 41, 50, 30, 39, 23,  2, 30,\n",
       "                   17, 30, 30,  7, 45, 30, 30, 22, 36,  9, 11, 32, 50, 64,\n",
       "                   19, 30, 33,  8, 17, 27, 30, 22, 22, 62, 48, 30, 39, 36,\n",
       "                   30, 40, 28, 30, 30, 24, 19, 29, 30, 32, 62, 53, 36, 30,\n",
       "                   16, 19, 34, 39, 30, 32, 25, 39, 54, 36, 30, 18, 47, 60,\n",
       "                   22, 30, 35, 52, 47, 30, 37, 36, 30, 49, 30, 49, 24, 30,\n",
       "                   30, 44, 35, 36, 30, 27, 22, 40, 39, 30, 30, 30, 35, 24,\n",
       "                   34, 26,  4, 26, 27, 42, 20, 21, 21, 61, 57, 21, 26, 30,\n",
       "                   71, 51, 32, 30,  9, 28, 32, 31, 41, 30, 20, 24,  2, 30,\n",
       "                    1, 48, 19, 56, 30, 23, 30, 18, 21, 30, 18, 24, 30, 32,\n",
       "                   23, 58, 50, 40, 47, 36, 20, 32, 25, 30, 43, 30, 40, 31,\n",
       "                   70, 31, 30, 18, 25, 18, 43, 36, 30, 27, 20, 14, 60, 25,\n",
       "                   14, 19, 18, 15, 31,  4, 30, 25, 60, 52, 44, 30, 49, 42,\n",
       "                   18, 35, 18, 25, 26, 39, 45, 42, 22, 30, 24, 30, 48, 29,\n",
       "                   52, 19, 38, 27, 30, 33,  6, 17, 34, 50, 27, 20, 30, 30,\n",
       "                   25, 25, 29, 11, 30, 23, 23, 29, 48, 35, 30, 30, 30, 36,\n",
       "                   21, 24, 31, 70, 16, 30, 19, 31,  4,  6, 33, 23, 48,  1,\n",
       "                   28, 18, 34, 33, 30, 41, 20, 36, 16, 51, 30, 31, 30, 32,\n",
       "                   24, 48, 57, 30, 54, 18, 30,  5, 30, 43, 13, 17, 29, 30,\n",
       "                   25, 25, 18,  8,  1, 46, 30, 16, 30, 30, 25, 39, 49, 31,\n",
       "                   30, 30, 34, 31, 11,  1, 27, 31, 39, 18, 39, 33, 26, 39,\n",
       "                   35,  6, 31, 30, 23, 31, 43, 10, 52, 27, 38, 27,  2, 30,\n",
       "                   30,  1, 30, 62, 15,  1, 30, 23, 18, 39, 21, 30, 32, 30,\n",
       "                   20, 16, 30, 35, 17, 42, 30, 35, 28, 30,  4, 71,  9, 16,\n",
       "                   44, 18, 45, 51, 24, 30, 41, 21, 48, 30, 24, 42, 27, 31,\n",
       "                   30,  4, 26, 47, 33, 47, 28, 15, 20, 19, 30, 56, 25, 33,\n",
       "                   22, 28, 25, 39, 27, 19, 30, 26, 32],\n",
       "             mask=False,\n",
       "       fill_value=999999,\n",
       "            dtype=int64)"
      ]
     },
     "execution_count": 183,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "winsorize(df[\"age\"], (0, 0.005))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 184,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x15008dca6a0>"
      ]
     },
     "execution_count": 184,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.boxplot(df[\"age\"])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 185,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x15008e215f8>"
      ]
     },
     "execution_count": 185,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.boxplot(x=pd.Series(winsorize(df[\"age\"], (0, 0.005))))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 186,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0      22\n",
       "1      38\n",
       "2      26\n",
       "3      35\n",
       "4      35\n",
       "5      30\n",
       "6      54\n",
       "7       2\n",
       "8      27\n",
       "9      14\n",
       "10      4\n",
       "11     58\n",
       "12     20\n",
       "13     39\n",
       "14     14\n",
       "15     55\n",
       "16      2\n",
       "17     30\n",
       "18     31\n",
       "19     30\n",
       "20     35\n",
       "21     34\n",
       "22     15\n",
       "23     28\n",
       "24      8\n",
       "25     38\n",
       "26     30\n",
       "27     19\n",
       "28     30\n",
       "29     30\n",
       "       ..\n",
       "861    21\n",
       "862    48\n",
       "863    30\n",
       "864    24\n",
       "865    42\n",
       "866    27\n",
       "867    31\n",
       "868    30\n",
       "869     4\n",
       "870    26\n",
       "871    47\n",
       "872    33\n",
       "873    47\n",
       "874    28\n",
       "875    15\n",
       "876    20\n",
       "877    19\n",
       "878    30\n",
       "879    56\n",
       "880    25\n",
       "881    33\n",
       "882    22\n",
       "883    28\n",
       "884    25\n",
       "885    39\n",
       "886    27\n",
       "887    19\n",
       "888    30\n",
       "889    26\n",
       "890    32\n",
       "Length: 891, dtype: int64"
      ]
     },
     "execution_count": 186,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "pd.Series(winsorize(df[\"age\"], (0, 0.005)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 187,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "96     71\n",
       "851    71\n",
       "630    71\n",
       "116    71\n",
       "493    71\n",
       "672    70\n",
       "745    70\n",
       "33     66\n",
       "280    65\n",
       "456    65\n",
       "54     65\n",
       "545    64\n",
       "438    64\n",
       "275    63\n",
       "483    63\n",
       "555    62\n",
       "570    62\n",
       "252    62\n",
       "829    62\n",
       "326    61\n",
       "170    61\n",
       "625    61\n",
       "366    60\n",
       "684    60\n",
       "694    60\n",
       "587    60\n",
       "94     59\n",
       "232    59\n",
       "268    58\n",
       "659    58\n",
       "       ..\n",
       "261     3\n",
       "348     3\n",
       "374     3\n",
       "43      3\n",
       "407     3\n",
       "193     3\n",
       "340     2\n",
       "297     2\n",
       "479     2\n",
       "205     2\n",
       "7       2\n",
       "530     2\n",
       "824     2\n",
       "119     2\n",
       "16      2\n",
       "642     2\n",
       "305     1\n",
       "78      1\n",
       "183     1\n",
       "831     1\n",
       "381     1\n",
       "386     1\n",
       "827     1\n",
       "172     1\n",
       "803     1\n",
       "788     1\n",
       "469     1\n",
       "755     1\n",
       "644     1\n",
       "164     1\n",
       "Length: 891, dtype: int64"
      ]
     },
     "execution_count": 187,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "pd.Series(winsorize(df[\"age\"], (0, 0.005))).sort_values(ascending=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 188,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_age_log = np.log(df[\"age\"])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 191,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0    3.091042\n",
       "1    3.637586\n",
       "2    3.258097\n",
       "3    3.555348\n",
       "4    3.555348\n",
       "Name: age, dtype: float64"
      ]
     },
     "execution_count": 191,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_age_log.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 192,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x15008e7d4a8>"
      ]
     },
     "execution_count": 192,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.boxplot(df_age_log)\n",
    "# applying log transformation is not a good idea for this case.SThus we should apply winsorize method or drop outliers"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 194,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "22.0"
      ]
     },
     "execution_count": 194,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "q1 = df[\"age\"].quantile(0.25)\n",
    "q1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 193,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "35.0"
      ]
     },
     "execution_count": 193,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "q3 = df[\"age\"].quantile(0.75)\n",
    "q3"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 195,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "13.0"
      ]
     },
     "execution_count": 195,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "iqr =q3 - q1\n",
    "iqr"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 196,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "54.5"
      ]
     },
     "execution_count": 196,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "ub = df[\"age\"].quantile(.75) + 1.5 * iqr\n",
    "ub"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 197,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "2.5"
      ]
     },
     "execution_count": 197,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "lb = df[\"age\"].quantile(.25) - 1.5 * iqr\n",
    "lb"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 198,
   "metadata": {},
   "outputs": [],
   "source": [
    "outliers_low = df[\"age\"] < lb"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 199,
   "metadata": {},
   "outputs": [],
   "source": [
    "outliers_up = df[\"age\"] > ub"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 201,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "7      2\n",
       "16     2\n",
       "78     1\n",
       "119    2\n",
       "164    1\n",
       "172    1\n",
       "183    1\n",
       "205    2\n",
       "297    2\n",
       "305    1\n",
       "340    2\n",
       "381    1\n",
       "386    1\n",
       "469    1\n",
       "479    2\n",
       "530    2\n",
       "642    2\n",
       "644    1\n",
       "755    1\n",
       "788    1\n",
       "803    1\n",
       "824    2\n",
       "827    1\n",
       "831    1\n",
       "Name: age, dtype: int64"
      ]
     },
     "execution_count": 201,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df[\"age\"][outliers_low]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 202,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "11     58\n",
       "15     55\n",
       "33     66\n",
       "54     65\n",
       "94     59\n",
       "96     71\n",
       "116    71\n",
       "152    56\n",
       "170    61\n",
       "174    56\n",
       "195    58\n",
       "232    59\n",
       "252    62\n",
       "268    58\n",
       "275    63\n",
       "280    65\n",
       "326    61\n",
       "366    60\n",
       "438    64\n",
       "456    65\n",
       "467    56\n",
       "483    63\n",
       "487    58\n",
       "492    55\n",
       "493    71\n",
       "545    64\n",
       "555    62\n",
       "570    62\n",
       "587    60\n",
       "625    61\n",
       "626    57\n",
       "630    80\n",
       "647    56\n",
       "659    58\n",
       "672    70\n",
       "684    60\n",
       "694    60\n",
       "745    70\n",
       "772    57\n",
       "829    62\n",
       "851    74\n",
       "879    56\n",
       "Name: age, dtype: int64"
      ]
     },
     "execution_count": 202,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df[\"age\"][outliers_up]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 203,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "825"
      ]
     },
     "execution_count": 203,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "len(df[\"age\"]) - (len(df[\"age\"][outliers_low]) + len(df[\"age\"][outliers_up]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 205,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "7       2\n",
       "11     58\n",
       "15     55\n",
       "16      2\n",
       "33     66\n",
       "54     65\n",
       "78      1\n",
       "94     59\n",
       "96     71\n",
       "116    71\n",
       "119     2\n",
       "152    56\n",
       "164     1\n",
       "170    61\n",
       "172     1\n",
       "174    56\n",
       "183     1\n",
       "195    58\n",
       "205     2\n",
       "232    59\n",
       "252    62\n",
       "268    58\n",
       "275    63\n",
       "280    65\n",
       "297     2\n",
       "305     1\n",
       "326    61\n",
       "340     2\n",
       "366    60\n",
       "381     1\n",
       "       ..\n",
       "483    63\n",
       "487    58\n",
       "492    55\n",
       "493    71\n",
       "530     2\n",
       "545    64\n",
       "555    62\n",
       "570    62\n",
       "587    60\n",
       "625    61\n",
       "626    57\n",
       "630    80\n",
       "642     2\n",
       "644     1\n",
       "647    56\n",
       "659    58\n",
       "672    70\n",
       "684    60\n",
       "694    60\n",
       "745    70\n",
       "755     1\n",
       "772    57\n",
       "788     1\n",
       "803     1\n",
       "824     2\n",
       "827     1\n",
       "829    62\n",
       "831     1\n",
       "851    74\n",
       "879    56\n",
       "Name: age, Length: 66, dtype: int64"
      ]
     },
     "execution_count": 205,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df[\"age\"][outliers_low | outliers_up]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 206,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0      22\n",
       "1      38\n",
       "2      26\n",
       "3      35\n",
       "4      35\n",
       "5      30\n",
       "6      54\n",
       "8      27\n",
       "9      14\n",
       "10      4\n",
       "12     20\n",
       "13     39\n",
       "14     14\n",
       "17     30\n",
       "18     31\n",
       "19     30\n",
       "20     35\n",
       "21     34\n",
       "22     15\n",
       "23     28\n",
       "24      8\n",
       "25     38\n",
       "26     30\n",
       "27     19\n",
       "28     30\n",
       "29     30\n",
       "30     40\n",
       "31     30\n",
       "32     30\n",
       "34     28\n",
       "       ..\n",
       "860    41\n",
       "861    21\n",
       "862    48\n",
       "863    30\n",
       "864    24\n",
       "865    42\n",
       "866    27\n",
       "867    31\n",
       "868    30\n",
       "869     4\n",
       "870    26\n",
       "871    47\n",
       "872    33\n",
       "873    47\n",
       "874    28\n",
       "875    15\n",
       "876    20\n",
       "877    19\n",
       "878    30\n",
       "880    25\n",
       "881    33\n",
       "882    22\n",
       "883    28\n",
       "884    25\n",
       "885    39\n",
       "886    27\n",
       "887    19\n",
       "888    30\n",
       "889    26\n",
       "890    32\n",
       "Name: age, Length: 825, dtype: int64"
      ]
     },
     "execution_count": 206,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df[\"age\"][~(outliers_low | outliers_up)]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 207,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(891, 14)"
      ]
     },
     "execution_count": 207,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 208,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = df[~(outliers_low | outliers_up)]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 209,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "Int64Index: 825 entries, 0 to 890\n",
      "Data columns (total 14 columns):\n",
      "survived       825 non-null int64\n",
      "pclass         825 non-null int64\n",
      "sex            825 non-null object\n",
      "age            825 non-null int64\n",
      "sibsp          825 non-null int64\n",
      "parch          825 non-null int64\n",
      "fare           825 non-null float64\n",
      "class          825 non-null category\n",
      "who            825 non-null object\n",
      "adult_male     825 non-null int32\n",
      "deck           825 non-null object\n",
      "embark_town    825 non-null object\n",
      "alive          825 non-null object\n",
      "alone          825 non-null int32\n",
      "dtypes: category(1), float64(1), int32(2), int64(5), object(5)\n",
      "memory usage: 84.7+ KB\n"
     ]
    }
   ],
   "source": [
    "df.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 211,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x15008f30da0>"
      ]
     },
     "execution_count": 211,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.boxplot(df[\"age\"]) # now there is a few outliers"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 212,
   "metadata": {},
   "outputs": [],
   "source": [
    "#df.rename( {\"deck\":\"guverte\", \"who\":\"man\"},axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 213,
   "metadata": {},
   "outputs": [],
   "source": [
    "#df[\"alive\"].dtypes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 214,
   "metadata": {},
   "outputs": [],
   "source": [
    "#df[\"alive\"].map({\"no\":0, \"yes\":1})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 215,
   "metadata": {},
   "outputs": [],
   "source": [
    "#df[\"alive\"].replace({\"yes\":1, \"no\":0})"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
