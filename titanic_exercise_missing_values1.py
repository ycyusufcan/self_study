{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 289,
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
   "execution_count": 290,
   "metadata": {},
   "outputs": [],
   "source": [
    "import ssl\n",
    "ssl._create_default_https_context = ssl._create_unverified_context"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 291,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = sns.load_dataset(\"titanic\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 292,
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
     "execution_count": 292,
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
   "execution_count": 293,
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
   "execution_count": 294,
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
     "execution_count": 294,
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
   "execution_count": 295,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(891, 15)"
      ]
     },
     "execution_count": 295,
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
   "execution_count": 296,
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
     "execution_count": 296,
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
   "execution_count": 297,
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
     "execution_count": 297,
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
   "execution_count": 298,
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
     "execution_count": 298,
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
   "execution_count": 299,
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
     "execution_count": 299,
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
   "execution_count": 300,
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
     "execution_count": 300,
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
   "execution_count": 301,
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
     "execution_count": 301,
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
   "execution_count": 302,
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
     "execution_count": 302,
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
   "execution_count": 303,
   "metadata": {},
   "outputs": [],
   "source": [
    "df.loc[df[(df[\"fare\"] < 10) & (df[\"deck\"].isnull())].index, \"deck\"] = \"F\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 304,
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
     "execution_count": 304,
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
   "execution_count": 305,
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
     "execution_count": 305,
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
   "execution_count": 306,
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
     "execution_count": 306,
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
   "execution_count": 307,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "7.8542"
      ]
     },
     "execution_count": 307,
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
   "execution_count": 308,
   "metadata": {},
   "outputs": [],
   "source": [
    "df.loc[df[(df[\"fare\"]==0) & (df[\"deck\"]==\"F\")].index, \"fare\"] = med_F"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 309,
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
     "execution_count": 309,
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
   "execution_count": 310,
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
     "execution_count": 310,
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
   "execution_count": 311,
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
     "execution_count": 311,
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
   "execution_count": 312,
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
     "execution_count": 312,
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
   "execution_count": 313,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "35.5"
      ]
     },
     "execution_count": 313,
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
   "execution_count": 314,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "35.5"
      ]
     },
     "execution_count": 314,
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
   "execution_count": 315,
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
     "execution_count": 315,
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
   "execution_count": 316,
   "metadata": {},
   "outputs": [],
   "source": [
    "df.loc[df[(df[\"deck\"]==\"A\") & (df[\"fare\"]==0)].index, \"fare\"] = df[df[\"deck\"]==\"A\"][\"fare\"].median()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 317,
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
     "execution_count": 317,
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
   "execution_count": 318,
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
     "execution_count": 318,
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
   "execution_count": 319,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "80.0"
      ]
     },
     "execution_count": 319,
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
   "execution_count": 320,
   "metadata": {},
   "outputs": [],
   "source": [
    "df.loc[df[(df[\"deck\"]==\"B\") & (df[\"fare\"]==0)].index, \"fare\"] = df.groupby(\"deck\")[\"fare\"].median()[\"B\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 321,
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
     "execution_count": 321,
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
   "execution_count": 322,
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
     "execution_count": 322,
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
   "execution_count": 323,
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
     "execution_count": 323,
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
   "execution_count": 324,
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
     "execution_count": 324,
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
   "execution_count": 325,
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
     "execution_count": 325,
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
   "execution_count": 326,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Let's share the remaining nan values proportionally to the other \"deck classes\" except \"F\"\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 327,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[340, 59, 47, 33, 32, 15, 4]"
      ]
     },
     "execution_count": 327,
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
   "execution_count": 328,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[59, 47, 33, 32, 15, 4]"
      ]
     },
     "execution_count": 328,
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
   "execution_count": 329,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "190"
      ]
     },
     "execution_count": 329,
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
   "execution_count": 330,
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
     "execution_count": 330,
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
   "execution_count": 331,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[112, 89, 63, 61, 28, 8]"
      ]
     },
     "execution_count": 331,
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
   "execution_count": 332,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "361"
      ]
     },
     "execution_count": 332,
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
   "execution_count": 333,
   "metadata": {},
   "outputs": [],
   "source": [
    "deck_names = df[\"deck\"].value_counts().to_dict()\n",
    "deck_names = list(deck_names.keys())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 334,
   "metadata": {},
   "outputs": [],
   "source": [
    "deck_names.remove(deck_names[0])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 335,
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
   "execution_count": 336,
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
     "execution_count": 336,
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
   "execution_count": 337,
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
     "execution_count": 337,
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
   "execution_count": 338,
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
     "execution_count": 338,
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
   "execution_count": 339,
   "metadata": {},
   "outputs": [],
   "source": [
    "df.drop(\"embarked\", axis=1, inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 340,
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
     "execution_count": 340,
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
   "execution_count": 341,
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
     "execution_count": 341,
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
   "execution_count": 342,
   "metadata": {},
   "outputs": [],
   "source": [
    "#let's fill nan values with most freqeunt\n",
    "df[\"embark_town\"].fillna(\"Southampton\", inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 343,
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
     "execution_count": 343,
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
   "execution_count": 344,
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
     "execution_count": 344,
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
   "execution_count": 345,
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
     "execution_count": 345,
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
   "execution_count": 346,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "29.69911764705882"
      ]
     },
     "execution_count": 346,
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
   "execution_count": 347,
   "metadata": {},
   "outputs": [],
   "source": [
    "df[\"age\"].fillna(df[\"age\"].mean(), inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 348,
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
     "execution_count": 348,
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
   "execution_count": 355,
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
     "execution_count": 355,
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
   "execution_count": 356,
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
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>male</td>\n",
       "      <td>22</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>7.2500</td>\n",
       "      <td>Third</td>\n",
       "      <td>man</td>\n",
       "      <td>True</td>\n",
       "      <td>F</td>\n",
       "      <td>Southampton</td>\n",
       "      <td>no</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>female</td>\n",
       "      <td>38</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>71.2833</td>\n",
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
       "      <td>26</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7.9250</td>\n",
       "      <td>Third</td>\n",
       "      <td>woman</td>\n",
       "      <td>False</td>\n",
       "      <td>F</td>\n",
       "      <td>Southampton</td>\n",
       "      <td>yes</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>female</td>\n",
       "      <td>35</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>53.1000</td>\n",
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
       "      <td>35</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>8.0500</td>\n",
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
       "   survived  pclass     sex  age  sibsp  parch     fare  class    who  \\\n",
       "0         0       3    male   22      1      0   7.2500  Third    man   \n",
       "1         1       1  female   38      1      0  71.2833  First  woman   \n",
       "2         1       3  female   26      0      0   7.9250  Third  woman   \n",
       "3         1       1  female   35      1      0  53.1000  First  woman   \n",
       "4         0       3    male   35      0      0   8.0500  Third    man   \n",
       "\n",
       "   adult_male deck  embark_town alive  alone  \n",
       "0        True    F  Southampton    no  False  \n",
       "1       False    C    Cherbourg   yes  False  \n",
       "2       False    F  Southampton   yes   True  \n",
       "3       False    C  Southampton   yes  False  \n",
       "4        True    F  Southampton    no   True  "
      ]
     },
     "execution_count": 356,
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
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
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
