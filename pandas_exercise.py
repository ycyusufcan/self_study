{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd \n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Creating a Series by passing a list of values, letting pandas create a default integer index"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0     1.0\n",
       "1     2.0\n",
       "2     3.0\n",
       "3     NaN\n",
       "4    45.0\n",
       "dtype: float64"
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "seri = pd.Series([1,2,3,np.nan,45])\n",
    "seri"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Creating a DataFrameby passing a numpy array, with a datetime index and labeled columns:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "DatetimeIndex(['2021-01-01', '2021-01-02', '2021-01-03', '2021-01-04',\n",
       "               '2021-01-05', '2021-01-06', '2021-01-07', '2021-01-08',\n",
       "               '2021-01-09', '2021-01-10'],\n",
       "              dtype='datetime64[ns]', freq='D')"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dates = pd.date_range(\"20210101\", periods=10)\n",
    "dates"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
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
       "      <th>a</th>\n",
       "      <th>b</th>\n",
       "      <th>c</th>\n",
       "      <th>d</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>2021-01-01</th>\n",
       "      <td>0.482631</td>\n",
       "      <td>-0.917418</td>\n",
       "      <td>0.652312</td>\n",
       "      <td>-0.675364</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-02</th>\n",
       "      <td>-0.601569</td>\n",
       "      <td>0.888478</td>\n",
       "      <td>0.875262</td>\n",
       "      <td>-1.645763</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-03</th>\n",
       "      <td>0.971676</td>\n",
       "      <td>0.167080</td>\n",
       "      <td>-1.057748</td>\n",
       "      <td>-1.223418</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-04</th>\n",
       "      <td>-0.424823</td>\n",
       "      <td>1.512963</td>\n",
       "      <td>-1.044807</td>\n",
       "      <td>0.357849</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-05</th>\n",
       "      <td>0.425621</td>\n",
       "      <td>-0.049688</td>\n",
       "      <td>-0.433479</td>\n",
       "      <td>0.120930</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-06</th>\n",
       "      <td>1.357304</td>\n",
       "      <td>0.178053</td>\n",
       "      <td>-1.265542</td>\n",
       "      <td>-0.643415</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-07</th>\n",
       "      <td>0.031349</td>\n",
       "      <td>-0.749075</td>\n",
       "      <td>-0.018996</td>\n",
       "      <td>0.386057</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-08</th>\n",
       "      <td>-0.882429</td>\n",
       "      <td>-0.817032</td>\n",
       "      <td>-1.165577</td>\n",
       "      <td>0.179282</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-09</th>\n",
       "      <td>1.548553</td>\n",
       "      <td>0.545712</td>\n",
       "      <td>-0.744566</td>\n",
       "      <td>1.493493</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-10</th>\n",
       "      <td>0.733131</td>\n",
       "      <td>0.771151</td>\n",
       "      <td>-0.131902</td>\n",
       "      <td>-0.619149</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                   a         b         c         d\n",
       "2021-01-01  0.482631 -0.917418  0.652312 -0.675364\n",
       "2021-01-02 -0.601569  0.888478  0.875262 -1.645763\n",
       "2021-01-03  0.971676  0.167080 -1.057748 -1.223418\n",
       "2021-01-04 -0.424823  1.512963 -1.044807  0.357849\n",
       "2021-01-05  0.425621 -0.049688 -0.433479  0.120930\n",
       "2021-01-06  1.357304  0.178053 -1.265542 -0.643415\n",
       "2021-01-07  0.031349 -0.749075 -0.018996  0.386057\n",
       "2021-01-08 -0.882429 -0.817032 -1.165577  0.179282\n",
       "2021-01-09  1.548553  0.545712 -0.744566  1.493493\n",
       "2021-01-10  0.733131  0.771151 -0.131902 -0.619149"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df = pd.DataFrame(np.random.randn(10,4), index=dates, columns=[\"a\",\"b\",\"c\",\"d\"]) #list('abcd') possible\n",
    "df"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Creating a DataFrame by passing a dict of objects that can be converted to series like"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
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
       "      <th>a</th>\n",
       "      <th>b</th>\n",
       "      <th>c</th>\n",
       "      <th>d</th>\n",
       "      <th>e</th>\n",
       "      <th>f</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>12</td>\n",
       "      <td>2020-12-01</td>\n",
       "      <td>1.0</td>\n",
       "      <td>3</td>\n",
       "      <td>test</td>\n",
       "      <td>foo</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>12</td>\n",
       "      <td>2020-12-01</td>\n",
       "      <td>1.0</td>\n",
       "      <td>3</td>\n",
       "      <td>train</td>\n",
       "      <td>foo</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>12</td>\n",
       "      <td>2020-12-01</td>\n",
       "      <td>1.0</td>\n",
       "      <td>3</td>\n",
       "      <td>test</td>\n",
       "      <td>foo</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>12</td>\n",
       "      <td>2020-12-01</td>\n",
       "      <td>1.0</td>\n",
       "      <td>3</td>\n",
       "      <td>train</td>\n",
       "      <td>foo</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "    a          b    c  d      e    f\n",
       "0  12 2020-12-01  1.0  3   test  foo\n",
       "1  12 2020-12-01  1.0  3  train  foo\n",
       "2  12 2020-12-01  1.0  3   test  foo\n",
       "3  12 2020-12-01  1.0  3  train  foo"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df2 = pd.DataFrame({\n",
    "    \"a\" : 12,\n",
    "    \"b\" : pd.Timestamp(\"20201201\"),\n",
    "    \"c\" : pd.Series(1, index=list(range(4)), dtype=float ),\n",
    "    \"d\" : np.array([3] * 4, dtype=\"int32\"),\n",
    "    \"e\" : pd.Categorical([\"test\", \"train\"]*2),\n",
    "    \"f\" : \"foo\"\n",
    "})\n",
    "df2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "a             int64\n",
       "b    datetime64[ns]\n",
       "c           float64\n",
       "d             int32\n",
       "e          category\n",
       "f            object\n",
       "dtype: object"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df2.dtypes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0   2020-12-01\n",
       "1   2020-12-01\n",
       "2   2020-12-01\n",
       "3   2020-12-01\n",
       "Name: b, dtype: datetime64[ns]"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df2.b"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Display the index, columns, and the underlying numpy data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "DatetimeIndex(['2021-01-01', '2021-01-02', '2021-01-03', '2021-01-04',\n",
       "               '2021-01-05', '2021-01-06', '2021-01-07', '2021-01-08',\n",
       "               '2021-01-09', '2021-01-10'],\n",
       "              dtype='datetime64[ns]', freq='D')"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.index"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['a', 'b', 'c', 'd'], dtype='object')"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ 0.48263137, -0.91741811,  0.65231161, -0.67536433],\n",
       "       [-0.60156895,  0.88847752,  0.87526232, -1.64576326],\n",
       "       [ 0.97167591,  0.16707971, -1.05774817, -1.22341839],\n",
       "       [-0.42482296,  1.5129627 , -1.04480731,  0.35784925],\n",
       "       [ 0.42562102, -0.04968797, -0.43347932,  0.12092959],\n",
       "       [ 1.35730385,  0.17805294, -1.26554199, -0.6434153 ],\n",
       "       [ 0.03134921, -0.74907544, -0.01899605,  0.38605705],\n",
       "       [-0.88242852, -0.817032  , -1.16557685,  0.17928164],\n",
       "       [ 1.54855298,  0.54571236, -0.74456577,  1.49349345],\n",
       "       [ 0.7331305 ,  0.77115105, -0.13190184, -0.61914928]])"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.values"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Describe shows a quick statistic summary of your data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
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
       "      <th>a</th>\n",
       "      <th>b</th>\n",
       "      <th>c</th>\n",
       "      <th>d</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>10.000000</td>\n",
       "      <td>10.000000</td>\n",
       "      <td>10.000000</td>\n",
       "      <td>10.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>0.364144</td>\n",
       "      <td>0.153022</td>\n",
       "      <td>-0.433504</td>\n",
       "      <td>-0.226950</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>0.825431</td>\n",
       "      <td>0.807545</td>\n",
       "      <td>0.763253</td>\n",
       "      <td>0.912753</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>-0.882429</td>\n",
       "      <td>-0.917418</td>\n",
       "      <td>-1.265542</td>\n",
       "      <td>-1.645763</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>-0.310780</td>\n",
       "      <td>-0.574229</td>\n",
       "      <td>-1.054513</td>\n",
       "      <td>-0.667377</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>0.454126</td>\n",
       "      <td>0.172566</td>\n",
       "      <td>-0.589023</td>\n",
       "      <td>-0.249110</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>0.912040</td>\n",
       "      <td>0.714791</td>\n",
       "      <td>-0.047222</td>\n",
       "      <td>0.313207</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>1.548553</td>\n",
       "      <td>1.512963</td>\n",
       "      <td>0.875262</td>\n",
       "      <td>1.493493</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "               a          b          c          d\n",
       "count  10.000000  10.000000  10.000000  10.000000\n",
       "mean    0.364144   0.153022  -0.433504  -0.226950\n",
       "std     0.825431   0.807545   0.763253   0.912753\n",
       "min    -0.882429  -0.917418  -1.265542  -1.645763\n",
       "25%    -0.310780  -0.574229  -1.054513  -0.667377\n",
       "50%     0.454126   0.172566  -0.589023  -0.249110\n",
       "75%     0.912040   0.714791  -0.047222   0.313207\n",
       "max     1.548553   1.512963   0.875262   1.493493"
      ]
     },
     "execution_count": 11,
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
   "execution_count": 12,
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
       "      <th>2021-01-01 00:00:00</th>\n",
       "      <th>2021-01-02 00:00:00</th>\n",
       "      <th>2021-01-03 00:00:00</th>\n",
       "      <th>2021-01-04 00:00:00</th>\n",
       "      <th>2021-01-05 00:00:00</th>\n",
       "      <th>2021-01-06 00:00:00</th>\n",
       "      <th>2021-01-07 00:00:00</th>\n",
       "      <th>2021-01-08 00:00:00</th>\n",
       "      <th>2021-01-09 00:00:00</th>\n",
       "      <th>2021-01-10 00:00:00</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>a</th>\n",
       "      <td>0.482631</td>\n",
       "      <td>-0.601569</td>\n",
       "      <td>0.971676</td>\n",
       "      <td>-0.424823</td>\n",
       "      <td>0.425621</td>\n",
       "      <td>1.357304</td>\n",
       "      <td>0.031349</td>\n",
       "      <td>-0.882429</td>\n",
       "      <td>1.548553</td>\n",
       "      <td>0.733131</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>b</th>\n",
       "      <td>-0.917418</td>\n",
       "      <td>0.888478</td>\n",
       "      <td>0.167080</td>\n",
       "      <td>1.512963</td>\n",
       "      <td>-0.049688</td>\n",
       "      <td>0.178053</td>\n",
       "      <td>-0.749075</td>\n",
       "      <td>-0.817032</td>\n",
       "      <td>0.545712</td>\n",
       "      <td>0.771151</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>c</th>\n",
       "      <td>0.652312</td>\n",
       "      <td>0.875262</td>\n",
       "      <td>-1.057748</td>\n",
       "      <td>-1.044807</td>\n",
       "      <td>-0.433479</td>\n",
       "      <td>-1.265542</td>\n",
       "      <td>-0.018996</td>\n",
       "      <td>-1.165577</td>\n",
       "      <td>-0.744566</td>\n",
       "      <td>-0.131902</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>d</th>\n",
       "      <td>-0.675364</td>\n",
       "      <td>-1.645763</td>\n",
       "      <td>-1.223418</td>\n",
       "      <td>0.357849</td>\n",
       "      <td>0.120930</td>\n",
       "      <td>-0.643415</td>\n",
       "      <td>0.386057</td>\n",
       "      <td>0.179282</td>\n",
       "      <td>1.493493</td>\n",
       "      <td>-0.619149</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   2021-01-01  2021-01-02  2021-01-03  2021-01-04  2021-01-05  2021-01-06  \\\n",
       "a    0.482631   -0.601569    0.971676   -0.424823    0.425621    1.357304   \n",
       "b   -0.917418    0.888478    0.167080    1.512963   -0.049688    0.178053   \n",
       "c    0.652312    0.875262   -1.057748   -1.044807   -0.433479   -1.265542   \n",
       "d   -0.675364   -1.645763   -1.223418    0.357849    0.120930   -0.643415   \n",
       "\n",
       "   2021-01-07  2021-01-08  2021-01-09  2021-01-10  \n",
       "a    0.031349   -0.882429    1.548553    0.733131  \n",
       "b   -0.749075   -0.817032    0.545712    0.771151  \n",
       "c   -0.018996   -1.165577   -0.744566   -0.131902  \n",
       "d    0.386057    0.179282    1.493493   -0.619149  "
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.T #transposing the data"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Sorting by an axis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
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
       "      <th>a</th>\n",
       "      <th>b</th>\n",
       "      <th>c</th>\n",
       "      <th>d</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>2021-01-01</th>\n",
       "      <td>0.482631</td>\n",
       "      <td>-0.917418</td>\n",
       "      <td>0.652312</td>\n",
       "      <td>-0.675364</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-02</th>\n",
       "      <td>-0.601569</td>\n",
       "      <td>0.888478</td>\n",
       "      <td>0.875262</td>\n",
       "      <td>-1.645763</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-03</th>\n",
       "      <td>0.971676</td>\n",
       "      <td>0.167080</td>\n",
       "      <td>-1.057748</td>\n",
       "      <td>-1.223418</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-04</th>\n",
       "      <td>-0.424823</td>\n",
       "      <td>1.512963</td>\n",
       "      <td>-1.044807</td>\n",
       "      <td>0.357849</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-05</th>\n",
       "      <td>0.425621</td>\n",
       "      <td>-0.049688</td>\n",
       "      <td>-0.433479</td>\n",
       "      <td>0.120930</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-06</th>\n",
       "      <td>1.357304</td>\n",
       "      <td>0.178053</td>\n",
       "      <td>-1.265542</td>\n",
       "      <td>-0.643415</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-07</th>\n",
       "      <td>0.031349</td>\n",
       "      <td>-0.749075</td>\n",
       "      <td>-0.018996</td>\n",
       "      <td>0.386057</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-08</th>\n",
       "      <td>-0.882429</td>\n",
       "      <td>-0.817032</td>\n",
       "      <td>-1.165577</td>\n",
       "      <td>0.179282</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-09</th>\n",
       "      <td>1.548553</td>\n",
       "      <td>0.545712</td>\n",
       "      <td>-0.744566</td>\n",
       "      <td>1.493493</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-10</th>\n",
       "      <td>0.733131</td>\n",
       "      <td>0.771151</td>\n",
       "      <td>-0.131902</td>\n",
       "      <td>-0.619149</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                   a         b         c         d\n",
       "2021-01-01  0.482631 -0.917418  0.652312 -0.675364\n",
       "2021-01-02 -0.601569  0.888478  0.875262 -1.645763\n",
       "2021-01-03  0.971676  0.167080 -1.057748 -1.223418\n",
       "2021-01-04 -0.424823  1.512963 -1.044807  0.357849\n",
       "2021-01-05  0.425621 -0.049688 -0.433479  0.120930\n",
       "2021-01-06  1.357304  0.178053 -1.265542 -0.643415\n",
       "2021-01-07  0.031349 -0.749075 -0.018996  0.386057\n",
       "2021-01-08 -0.882429 -0.817032 -1.165577  0.179282\n",
       "2021-01-09  1.548553  0.545712 -0.744566  1.493493\n",
       "2021-01-10  0.733131  0.771151 -0.131902 -0.619149"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.sort_index(axis=1, ascending=False)\n",
    "df"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Sorting by value"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
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
       "      <th>a</th>\n",
       "      <th>b</th>\n",
       "      <th>c</th>\n",
       "      <th>d</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>2021-01-06</th>\n",
       "      <td>1.357304</td>\n",
       "      <td>0.178053</td>\n",
       "      <td>-1.265542</td>\n",
       "      <td>-0.643415</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-08</th>\n",
       "      <td>-0.882429</td>\n",
       "      <td>-0.817032</td>\n",
       "      <td>-1.165577</td>\n",
       "      <td>0.179282</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-03</th>\n",
       "      <td>0.971676</td>\n",
       "      <td>0.167080</td>\n",
       "      <td>-1.057748</td>\n",
       "      <td>-1.223418</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-04</th>\n",
       "      <td>-0.424823</td>\n",
       "      <td>1.512963</td>\n",
       "      <td>-1.044807</td>\n",
       "      <td>0.357849</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-09</th>\n",
       "      <td>1.548553</td>\n",
       "      <td>0.545712</td>\n",
       "      <td>-0.744566</td>\n",
       "      <td>1.493493</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-05</th>\n",
       "      <td>0.425621</td>\n",
       "      <td>-0.049688</td>\n",
       "      <td>-0.433479</td>\n",
       "      <td>0.120930</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-10</th>\n",
       "      <td>0.733131</td>\n",
       "      <td>0.771151</td>\n",
       "      <td>-0.131902</td>\n",
       "      <td>-0.619149</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-07</th>\n",
       "      <td>0.031349</td>\n",
       "      <td>-0.749075</td>\n",
       "      <td>-0.018996</td>\n",
       "      <td>0.386057</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-01</th>\n",
       "      <td>0.482631</td>\n",
       "      <td>-0.917418</td>\n",
       "      <td>0.652312</td>\n",
       "      <td>-0.675364</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-02</th>\n",
       "      <td>-0.601569</td>\n",
       "      <td>0.888478</td>\n",
       "      <td>0.875262</td>\n",
       "      <td>-1.645763</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                   a         b         c         d\n",
       "2021-01-06  1.357304  0.178053 -1.265542 -0.643415\n",
       "2021-01-08 -0.882429 -0.817032 -1.165577  0.179282\n",
       "2021-01-03  0.971676  0.167080 -1.057748 -1.223418\n",
       "2021-01-04 -0.424823  1.512963 -1.044807  0.357849\n",
       "2021-01-09  1.548553  0.545712 -0.744566  1.493493\n",
       "2021-01-05  0.425621 -0.049688 -0.433479  0.120930\n",
       "2021-01-10  0.733131  0.771151 -0.131902 -0.619149\n",
       "2021-01-07  0.031349 -0.749075 -0.018996  0.386057\n",
       "2021-01-01  0.482631 -0.917418  0.652312 -0.675364\n",
       "2021-01-02 -0.601569  0.888478  0.875262 -1.645763"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.sort_values(by=\"c\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Selection\n",
    "Getting"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "2021-01-01    0.482631\n",
       "2021-01-02   -0.601569\n",
       "2021-01-03    0.971676\n",
       "2021-01-04   -0.424823\n",
       "2021-01-05    0.425621\n",
       "2021-01-06    1.357304\n",
       "2021-01-07    0.031349\n",
       "2021-01-08   -0.882429\n",
       "2021-01-09    1.548553\n",
       "2021-01-10    0.733131\n",
       "Freq: D, Name: a, dtype: float64"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df[\"a\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "2021-01-01    0.482631\n",
       "2021-01-02   -0.601569\n",
       "2021-01-03    0.971676\n",
       "2021-01-04   -0.424823\n",
       "2021-01-05    0.425621\n",
       "2021-01-06    1.357304\n",
       "2021-01-07    0.031349\n",
       "2021-01-08   -0.882429\n",
       "2021-01-09    1.548553\n",
       "2021-01-10    0.733131\n",
       "Freq: D, Name: a, dtype: float64"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.a"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Selecting via [], which slices the rows"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
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
       "      <th>a</th>\n",
       "      <th>b</th>\n",
       "      <th>c</th>\n",
       "      <th>d</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>2021-01-01</th>\n",
       "      <td>0.482631</td>\n",
       "      <td>-0.917418</td>\n",
       "      <td>0.652312</td>\n",
       "      <td>-0.675364</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-02</th>\n",
       "      <td>-0.601569</td>\n",
       "      <td>0.888478</td>\n",
       "      <td>0.875262</td>\n",
       "      <td>-1.645763</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-03</th>\n",
       "      <td>0.971676</td>\n",
       "      <td>0.167080</td>\n",
       "      <td>-1.057748</td>\n",
       "      <td>-1.223418</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-04</th>\n",
       "      <td>-0.424823</td>\n",
       "      <td>1.512963</td>\n",
       "      <td>-1.044807</td>\n",
       "      <td>0.357849</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                   a         b         c         d\n",
       "2021-01-01  0.482631 -0.917418  0.652312 -0.675364\n",
       "2021-01-02 -0.601569  0.888478  0.875262 -1.645763\n",
       "2021-01-03  0.971676  0.167080 -1.057748 -1.223418\n",
       "2021-01-04 -0.424823  1.512963 -1.044807  0.357849"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df[0:4]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
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
       "      <th>a</th>\n",
       "      <th>b</th>\n",
       "      <th>c</th>\n",
       "      <th>d</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>2021-01-02</th>\n",
       "      <td>-0.601569</td>\n",
       "      <td>0.888478</td>\n",
       "      <td>0.875262</td>\n",
       "      <td>-1.645763</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-03</th>\n",
       "      <td>0.971676</td>\n",
       "      <td>0.167080</td>\n",
       "      <td>-1.057748</td>\n",
       "      <td>-1.223418</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-04</th>\n",
       "      <td>-0.424823</td>\n",
       "      <td>1.512963</td>\n",
       "      <td>-1.044807</td>\n",
       "      <td>0.357849</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-05</th>\n",
       "      <td>0.425621</td>\n",
       "      <td>-0.049688</td>\n",
       "      <td>-0.433479</td>\n",
       "      <td>0.120930</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-06</th>\n",
       "      <td>1.357304</td>\n",
       "      <td>0.178053</td>\n",
       "      <td>-1.265542</td>\n",
       "      <td>-0.643415</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-07</th>\n",
       "      <td>0.031349</td>\n",
       "      <td>-0.749075</td>\n",
       "      <td>-0.018996</td>\n",
       "      <td>0.386057</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                   a         b         c         d\n",
       "2021-01-02 -0.601569  0.888478  0.875262 -1.645763\n",
       "2021-01-03  0.971676  0.167080 -1.057748 -1.223418\n",
       "2021-01-04 -0.424823  1.512963 -1.044807  0.357849\n",
       "2021-01-05  0.425621 -0.049688 -0.433479  0.120930\n",
       "2021-01-06  1.357304  0.178053 -1.265542 -0.643415\n",
       "2021-01-07  0.031349 -0.749075 -0.018996  0.386057"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df[\"20210102\":\"20210107\"]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Selection by Label"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "a    0.482631\n",
       "b   -0.917418\n",
       "c    0.652312\n",
       "d   -0.675364\n",
       "Name: 2021-01-01 00:00:00, dtype: float64"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.loc[dates[0]]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Selecting on a multi-axis by label"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
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
       "      <th>b</th>\n",
       "      <th>c</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>2021-01-01</th>\n",
       "      <td>-0.917418</td>\n",
       "      <td>0.652312</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-02</th>\n",
       "      <td>0.888478</td>\n",
       "      <td>0.875262</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-03</th>\n",
       "      <td>0.167080</td>\n",
       "      <td>-1.057748</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-04</th>\n",
       "      <td>1.512963</td>\n",
       "      <td>-1.044807</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-05</th>\n",
       "      <td>-0.049688</td>\n",
       "      <td>-0.433479</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-06</th>\n",
       "      <td>0.178053</td>\n",
       "      <td>-1.265542</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-07</th>\n",
       "      <td>-0.749075</td>\n",
       "      <td>-0.018996</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-08</th>\n",
       "      <td>-0.817032</td>\n",
       "      <td>-1.165577</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-09</th>\n",
       "      <td>0.545712</td>\n",
       "      <td>-0.744566</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-10</th>\n",
       "      <td>0.771151</td>\n",
       "      <td>-0.131902</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                   b         c\n",
       "2021-01-01 -0.917418  0.652312\n",
       "2021-01-02  0.888478  0.875262\n",
       "2021-01-03  0.167080 -1.057748\n",
       "2021-01-04  1.512963 -1.044807\n",
       "2021-01-05 -0.049688 -0.433479\n",
       "2021-01-06  0.178053 -1.265542\n",
       "2021-01-07 -0.749075 -0.018996\n",
       "2021-01-08 -0.817032 -1.165577\n",
       "2021-01-09  0.545712 -0.744566\n",
       "2021-01-10  0.771151 -0.131902"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.loc[:,[\"b\",\"c\"]]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Showing label slicing, both endpoints are included"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
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
       "      <th>a</th>\n",
       "      <th>b</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>2021-01-03</th>\n",
       "      <td>0.971676</td>\n",
       "      <td>0.167080</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-04</th>\n",
       "      <td>-0.424823</td>\n",
       "      <td>1.512963</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-05</th>\n",
       "      <td>0.425621</td>\n",
       "      <td>-0.049688</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-06</th>\n",
       "      <td>1.357304</td>\n",
       "      <td>0.178053</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                   a         b\n",
       "2021-01-03  0.971676  0.167080\n",
       "2021-01-04 -0.424823  1.512963\n",
       "2021-01-05  0.425621 -0.049688\n",
       "2021-01-06  1.357304  0.178053"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.loc[\"20210103\":\"20210106\",[\"a\",\"b\"]]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Reduction in the dimensions of the returned object"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "c   -1.057748\n",
       "d   -1.223418\n",
       "Name: 2021-01-03 00:00:00, dtype: float64"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.loc[\"20210103\", [\"c\",\"d\"]]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "For getting a scalar value"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "-0.6015689504982479"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.loc[dates[1],\"a\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "-0.6015689504982479"
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.at[dates[1],\"a\"] #same with the previous method"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Selection by Position"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "a   -0.424823\n",
       "b    1.512963\n",
       "c   -1.044807\n",
       "d    0.357849\n",
       "Name: 2021-01-04 00:00:00, dtype: float64"
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.iloc[3]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "By integer slices, acting similar to numpy/python"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
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
       "      <th>c</th>\n",
       "      <th>d</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>2021-01-04</th>\n",
       "      <td>-1.044807</td>\n",
       "      <td>0.357849</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-05</th>\n",
       "      <td>-0.433479</td>\n",
       "      <td>0.120930</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                   c         d\n",
       "2021-01-04 -1.044807  0.357849\n",
       "2021-01-05 -0.433479  0.120930"
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.iloc[3:5, 2:4]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "By lists of integer position locations, similar to the numpy/python style"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
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
       "      <th>a</th>\n",
       "      <th>c</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>2021-01-02</th>\n",
       "      <td>-0.601569</td>\n",
       "      <td>0.875262</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-03</th>\n",
       "      <td>0.971676</td>\n",
       "      <td>-1.057748</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-05</th>\n",
       "      <td>0.425621</td>\n",
       "      <td>-0.433479</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                   a         c\n",
       "2021-01-02 -0.601569  0.875262\n",
       "2021-01-03  0.971676 -1.057748\n",
       "2021-01-05  0.425621 -0.433479"
      ]
     },
     "execution_count": 27,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.iloc[[1,2,4],[0,2]]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "For slicing rows explicitly"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
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
       "      <th>a</th>\n",
       "      <th>b</th>\n",
       "      <th>c</th>\n",
       "      <th>d</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>2021-01-02</th>\n",
       "      <td>-0.601569</td>\n",
       "      <td>0.888478</td>\n",
       "      <td>0.875262</td>\n",
       "      <td>-1.645763</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-03</th>\n",
       "      <td>0.971676</td>\n",
       "      <td>0.167080</td>\n",
       "      <td>-1.057748</td>\n",
       "      <td>-1.223418</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                   a         b         c         d\n",
       "2021-01-02 -0.601569  0.888478  0.875262 -1.645763\n",
       "2021-01-03  0.971676  0.167080 -1.057748 -1.223418"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.iloc[1:3,:]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "For slicing columns explicitly"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
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
       "      <th>b</th>\n",
       "      <th>c</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>2021-01-01</th>\n",
       "      <td>-0.917418</td>\n",
       "      <td>0.652312</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-02</th>\n",
       "      <td>0.888478</td>\n",
       "      <td>0.875262</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-03</th>\n",
       "      <td>0.167080</td>\n",
       "      <td>-1.057748</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-04</th>\n",
       "      <td>1.512963</td>\n",
       "      <td>-1.044807</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-05</th>\n",
       "      <td>-0.049688</td>\n",
       "      <td>-0.433479</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-06</th>\n",
       "      <td>0.178053</td>\n",
       "      <td>-1.265542</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-07</th>\n",
       "      <td>-0.749075</td>\n",
       "      <td>-0.018996</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-08</th>\n",
       "      <td>-0.817032</td>\n",
       "      <td>-1.165577</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-09</th>\n",
       "      <td>0.545712</td>\n",
       "      <td>-0.744566</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-10</th>\n",
       "      <td>0.771151</td>\n",
       "      <td>-0.131902</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                   b         c\n",
       "2021-01-01 -0.917418  0.652312\n",
       "2021-01-02  0.888478  0.875262\n",
       "2021-01-03  0.167080 -1.057748\n",
       "2021-01-04  1.512963 -1.044807\n",
       "2021-01-05 -0.049688 -0.433479\n",
       "2021-01-06  0.178053 -1.265542\n",
       "2021-01-07 -0.749075 -0.018996\n",
       "2021-01-08 -0.817032 -1.165577\n",
       "2021-01-09  0.545712 -0.744566\n",
       "2021-01-10  0.771151 -0.131902"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.iloc[:,1:3]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "For getting a value explicitly"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.888477515439966"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.iloc[1,1]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.888477515439966"
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.iat[1,1] # same with the previous method"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Boolean Indexing\n",
    "\n",
    "Using a single column’s values to select data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
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
       "      <th>a</th>\n",
       "      <th>b</th>\n",
       "      <th>c</th>\n",
       "      <th>d</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>2021-01-01</th>\n",
       "      <td>0.482631</td>\n",
       "      <td>-0.917418</td>\n",
       "      <td>0.652312</td>\n",
       "      <td>-0.675364</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-03</th>\n",
       "      <td>0.971676</td>\n",
       "      <td>0.167080</td>\n",
       "      <td>-1.057748</td>\n",
       "      <td>-1.223418</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-05</th>\n",
       "      <td>0.425621</td>\n",
       "      <td>-0.049688</td>\n",
       "      <td>-0.433479</td>\n",
       "      <td>0.120930</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-06</th>\n",
       "      <td>1.357304</td>\n",
       "      <td>0.178053</td>\n",
       "      <td>-1.265542</td>\n",
       "      <td>-0.643415</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-07</th>\n",
       "      <td>0.031349</td>\n",
       "      <td>-0.749075</td>\n",
       "      <td>-0.018996</td>\n",
       "      <td>0.386057</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-09</th>\n",
       "      <td>1.548553</td>\n",
       "      <td>0.545712</td>\n",
       "      <td>-0.744566</td>\n",
       "      <td>1.493493</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-10</th>\n",
       "      <td>0.733131</td>\n",
       "      <td>0.771151</td>\n",
       "      <td>-0.131902</td>\n",
       "      <td>-0.619149</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                   a         b         c         d\n",
       "2021-01-01  0.482631 -0.917418  0.652312 -0.675364\n",
       "2021-01-03  0.971676  0.167080 -1.057748 -1.223418\n",
       "2021-01-05  0.425621 -0.049688 -0.433479  0.120930\n",
       "2021-01-06  1.357304  0.178053 -1.265542 -0.643415\n",
       "2021-01-07  0.031349 -0.749075 -0.018996  0.386057\n",
       "2021-01-09  1.548553  0.545712 -0.744566  1.493493\n",
       "2021-01-10  0.733131  0.771151 -0.131902 -0.619149"
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df[df.a > 0]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "A where operation for getting"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
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
       "      <th>a</th>\n",
       "      <th>b</th>\n",
       "      <th>c</th>\n",
       "      <th>d</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>2021-01-01</th>\n",
       "      <td>0.482631</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0.652312</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-02</th>\n",
       "      <td>NaN</td>\n",
       "      <td>0.888478</td>\n",
       "      <td>0.875262</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-03</th>\n",
       "      <td>0.971676</td>\n",
       "      <td>0.167080</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-04</th>\n",
       "      <td>NaN</td>\n",
       "      <td>1.512963</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0.357849</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-05</th>\n",
       "      <td>0.425621</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0.120930</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-06</th>\n",
       "      <td>1.357304</td>\n",
       "      <td>0.178053</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-07</th>\n",
       "      <td>0.031349</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0.386057</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-08</th>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0.179282</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-09</th>\n",
       "      <td>1.548553</td>\n",
       "      <td>0.545712</td>\n",
       "      <td>NaN</td>\n",
       "      <td>1.493493</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-10</th>\n",
       "      <td>0.733131</td>\n",
       "      <td>0.771151</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                   a         b         c         d\n",
       "2021-01-01  0.482631       NaN  0.652312       NaN\n",
       "2021-01-02       NaN  0.888478  0.875262       NaN\n",
       "2021-01-03  0.971676  0.167080       NaN       NaN\n",
       "2021-01-04       NaN  1.512963       NaN  0.357849\n",
       "2021-01-05  0.425621       NaN       NaN  0.120930\n",
       "2021-01-06  1.357304  0.178053       NaN       NaN\n",
       "2021-01-07  0.031349       NaN       NaN  0.386057\n",
       "2021-01-08       NaN       NaN       NaN  0.179282\n",
       "2021-01-09  1.548553  0.545712       NaN  1.493493\n",
       "2021-01-10  0.733131  0.771151       NaN       NaN"
      ]
     },
     "execution_count": 33,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df[df > 0]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Using the isin() method for filtering"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
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
       "      <th>a</th>\n",
       "      <th>b</th>\n",
       "      <th>c</th>\n",
       "      <th>d</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>2021-01-01</th>\n",
       "      <td>0.482631</td>\n",
       "      <td>-0.917418</td>\n",
       "      <td>0.652312</td>\n",
       "      <td>-0.675364</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-02</th>\n",
       "      <td>-0.601569</td>\n",
       "      <td>0.888478</td>\n",
       "      <td>0.875262</td>\n",
       "      <td>-1.645763</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-03</th>\n",
       "      <td>0.971676</td>\n",
       "      <td>0.167080</td>\n",
       "      <td>-1.057748</td>\n",
       "      <td>-1.223418</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-04</th>\n",
       "      <td>-0.424823</td>\n",
       "      <td>1.512963</td>\n",
       "      <td>-1.044807</td>\n",
       "      <td>0.357849</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-05</th>\n",
       "      <td>0.425621</td>\n",
       "      <td>-0.049688</td>\n",
       "      <td>-0.433479</td>\n",
       "      <td>0.120930</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-06</th>\n",
       "      <td>1.357304</td>\n",
       "      <td>0.178053</td>\n",
       "      <td>-1.265542</td>\n",
       "      <td>-0.643415</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-07</th>\n",
       "      <td>0.031349</td>\n",
       "      <td>-0.749075</td>\n",
       "      <td>-0.018996</td>\n",
       "      <td>0.386057</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-08</th>\n",
       "      <td>-0.882429</td>\n",
       "      <td>-0.817032</td>\n",
       "      <td>-1.165577</td>\n",
       "      <td>0.179282</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-09</th>\n",
       "      <td>1.548553</td>\n",
       "      <td>0.545712</td>\n",
       "      <td>-0.744566</td>\n",
       "      <td>1.493493</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-10</th>\n",
       "      <td>0.733131</td>\n",
       "      <td>0.771151</td>\n",
       "      <td>-0.131902</td>\n",
       "      <td>-0.619149</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                   a         b         c         d\n",
       "2021-01-01  0.482631 -0.917418  0.652312 -0.675364\n",
       "2021-01-02 -0.601569  0.888478  0.875262 -1.645763\n",
       "2021-01-03  0.971676  0.167080 -1.057748 -1.223418\n",
       "2021-01-04 -0.424823  1.512963 -1.044807  0.357849\n",
       "2021-01-05  0.425621 -0.049688 -0.433479  0.120930\n",
       "2021-01-06  1.357304  0.178053 -1.265542 -0.643415\n",
       "2021-01-07  0.031349 -0.749075 -0.018996  0.386057\n",
       "2021-01-08 -0.882429 -0.817032 -1.165577  0.179282\n",
       "2021-01-09  1.548553  0.545712 -0.744566  1.493493\n",
       "2021-01-10  0.733131  0.771151 -0.131902 -0.619149"
      ]
     },
     "execution_count": 34,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df2 = df.copy()\n",
    "df2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
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
       "      <th>a</th>\n",
       "      <th>b</th>\n",
       "      <th>c</th>\n",
       "      <th>d</th>\n",
       "      <th>e</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>2021-01-01</th>\n",
       "      <td>0.482631</td>\n",
       "      <td>-0.917418</td>\n",
       "      <td>0.652312</td>\n",
       "      <td>-0.675364</td>\n",
       "      <td>one</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-02</th>\n",
       "      <td>-0.601569</td>\n",
       "      <td>0.888478</td>\n",
       "      <td>0.875262</td>\n",
       "      <td>-1.645763</td>\n",
       "      <td>two</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-03</th>\n",
       "      <td>0.971676</td>\n",
       "      <td>0.167080</td>\n",
       "      <td>-1.057748</td>\n",
       "      <td>-1.223418</td>\n",
       "      <td>three</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-04</th>\n",
       "      <td>-0.424823</td>\n",
       "      <td>1.512963</td>\n",
       "      <td>-1.044807</td>\n",
       "      <td>0.357849</td>\n",
       "      <td>four</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-05</th>\n",
       "      <td>0.425621</td>\n",
       "      <td>-0.049688</td>\n",
       "      <td>-0.433479</td>\n",
       "      <td>0.120930</td>\n",
       "      <td>five</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-06</th>\n",
       "      <td>1.357304</td>\n",
       "      <td>0.178053</td>\n",
       "      <td>-1.265542</td>\n",
       "      <td>-0.643415</td>\n",
       "      <td>one</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-07</th>\n",
       "      <td>0.031349</td>\n",
       "      <td>-0.749075</td>\n",
       "      <td>-0.018996</td>\n",
       "      <td>0.386057</td>\n",
       "      <td>two</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-08</th>\n",
       "      <td>-0.882429</td>\n",
       "      <td>-0.817032</td>\n",
       "      <td>-1.165577</td>\n",
       "      <td>0.179282</td>\n",
       "      <td>three</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-09</th>\n",
       "      <td>1.548553</td>\n",
       "      <td>0.545712</td>\n",
       "      <td>-0.744566</td>\n",
       "      <td>1.493493</td>\n",
       "      <td>four</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-10</th>\n",
       "      <td>0.733131</td>\n",
       "      <td>0.771151</td>\n",
       "      <td>-0.131902</td>\n",
       "      <td>-0.619149</td>\n",
       "      <td>five</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                   a         b         c         d      e\n",
       "2021-01-01  0.482631 -0.917418  0.652312 -0.675364    one\n",
       "2021-01-02 -0.601569  0.888478  0.875262 -1.645763    two\n",
       "2021-01-03  0.971676  0.167080 -1.057748 -1.223418  three\n",
       "2021-01-04 -0.424823  1.512963 -1.044807  0.357849   four\n",
       "2021-01-05  0.425621 -0.049688 -0.433479  0.120930   five\n",
       "2021-01-06  1.357304  0.178053 -1.265542 -0.643415    one\n",
       "2021-01-07  0.031349 -0.749075 -0.018996  0.386057    two\n",
       "2021-01-08 -0.882429 -0.817032 -1.165577  0.179282  three\n",
       "2021-01-09  1.548553  0.545712 -0.744566  1.493493   four\n",
       "2021-01-10  0.733131  0.771151 -0.131902 -0.619149   five"
      ]
     },
     "execution_count": 35,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df2[\"e\"] = [\"one\", \"two\", \"three\", \"four\", \"five\"]*2\n",
    "df2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
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
       "      <th>a</th>\n",
       "      <th>b</th>\n",
       "      <th>c</th>\n",
       "      <th>d</th>\n",
       "      <th>e</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>2021-01-01</th>\n",
       "      <td>0.482631</td>\n",
       "      <td>-0.917418</td>\n",
       "      <td>0.652312</td>\n",
       "      <td>-0.675364</td>\n",
       "      <td>one</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-02</th>\n",
       "      <td>-0.601569</td>\n",
       "      <td>0.888478</td>\n",
       "      <td>0.875262</td>\n",
       "      <td>-1.645763</td>\n",
       "      <td>two</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-06</th>\n",
       "      <td>1.357304</td>\n",
       "      <td>0.178053</td>\n",
       "      <td>-1.265542</td>\n",
       "      <td>-0.643415</td>\n",
       "      <td>one</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-07</th>\n",
       "      <td>0.031349</td>\n",
       "      <td>-0.749075</td>\n",
       "      <td>-0.018996</td>\n",
       "      <td>0.386057</td>\n",
       "      <td>two</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                   a         b         c         d    e\n",
       "2021-01-01  0.482631 -0.917418  0.652312 -0.675364  one\n",
       "2021-01-02 -0.601569  0.888478  0.875262 -1.645763  two\n",
       "2021-01-06  1.357304  0.178053 -1.265542 -0.643415  one\n",
       "2021-01-07  0.031349 -0.749075 -0.018996  0.386057  two"
      ]
     },
     "execution_count": 36,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df2[df2[\"e\"].isin([\"one\", \"two\"])]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Setting\n",
    "Setting a new column automatically aligns the data by the indexes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "2021-01-01    1\n",
       "2021-01-02    2\n",
       "2021-01-03    3\n",
       "2021-01-04    4\n",
       "2021-01-05    5\n",
       "2021-01-06    6\n",
       "Freq: D, dtype: int32"
      ]
     },
     "execution_count": 37,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "s1 = pd.Series(np.arange(1,7), index=pd.date_range(\"20210101\", periods=6))\n",
    "s1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
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
       "      <th>a</th>\n",
       "      <th>b</th>\n",
       "      <th>c</th>\n",
       "      <th>d</th>\n",
       "      <th>f</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>2021-01-01</th>\n",
       "      <td>0.482631</td>\n",
       "      <td>-0.917418</td>\n",
       "      <td>0.652312</td>\n",
       "      <td>-0.675364</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-02</th>\n",
       "      <td>-0.601569</td>\n",
       "      <td>0.888478</td>\n",
       "      <td>0.875262</td>\n",
       "      <td>-1.645763</td>\n",
       "      <td>2.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-03</th>\n",
       "      <td>0.971676</td>\n",
       "      <td>0.167080</td>\n",
       "      <td>-1.057748</td>\n",
       "      <td>-1.223418</td>\n",
       "      <td>3.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-04</th>\n",
       "      <td>-0.424823</td>\n",
       "      <td>1.512963</td>\n",
       "      <td>-1.044807</td>\n",
       "      <td>0.357849</td>\n",
       "      <td>4.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-05</th>\n",
       "      <td>0.425621</td>\n",
       "      <td>-0.049688</td>\n",
       "      <td>-0.433479</td>\n",
       "      <td>0.120930</td>\n",
       "      <td>5.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-06</th>\n",
       "      <td>1.357304</td>\n",
       "      <td>0.178053</td>\n",
       "      <td>-1.265542</td>\n",
       "      <td>-0.643415</td>\n",
       "      <td>6.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-07</th>\n",
       "      <td>0.031349</td>\n",
       "      <td>-0.749075</td>\n",
       "      <td>-0.018996</td>\n",
       "      <td>0.386057</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-08</th>\n",
       "      <td>-0.882429</td>\n",
       "      <td>-0.817032</td>\n",
       "      <td>-1.165577</td>\n",
       "      <td>0.179282</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-09</th>\n",
       "      <td>1.548553</td>\n",
       "      <td>0.545712</td>\n",
       "      <td>-0.744566</td>\n",
       "      <td>1.493493</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-10</th>\n",
       "      <td>0.733131</td>\n",
       "      <td>0.771151</td>\n",
       "      <td>-0.131902</td>\n",
       "      <td>-0.619149</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                   a         b         c         d    f\n",
       "2021-01-01  0.482631 -0.917418  0.652312 -0.675364  1.0\n",
       "2021-01-02 -0.601569  0.888478  0.875262 -1.645763  2.0\n",
       "2021-01-03  0.971676  0.167080 -1.057748 -1.223418  3.0\n",
       "2021-01-04 -0.424823  1.512963 -1.044807  0.357849  4.0\n",
       "2021-01-05  0.425621 -0.049688 -0.433479  0.120930  5.0\n",
       "2021-01-06  1.357304  0.178053 -1.265542 -0.643415  6.0\n",
       "2021-01-07  0.031349 -0.749075 -0.018996  0.386057  NaN\n",
       "2021-01-08 -0.882429 -0.817032 -1.165577  0.179282  NaN\n",
       "2021-01-09  1.548553  0.545712 -0.744566  1.493493  NaN\n",
       "2021-01-10  0.733131  0.771151 -0.131902 -0.619149  NaN"
      ]
     },
     "execution_count": 38,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df[\"f\"] = s1\n",
    "df"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Setting values by label"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {},
   "outputs": [],
   "source": [
    "df.at[dates[0],\"a\"] = 0"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Setting values by position"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [],
   "source": [
    "df.iat[0,1] = 0"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Setting by assigning with a numpy array"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
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
       "      <th>a</th>\n",
       "      <th>b</th>\n",
       "      <th>c</th>\n",
       "      <th>d</th>\n",
       "      <th>f</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>2021-01-01</th>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.652312</td>\n",
       "      <td>5</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-02</th>\n",
       "      <td>-0.601569</td>\n",
       "      <td>0.888478</td>\n",
       "      <td>0.875262</td>\n",
       "      <td>5</td>\n",
       "      <td>2.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-03</th>\n",
       "      <td>0.971676</td>\n",
       "      <td>0.167080</td>\n",
       "      <td>-1.057748</td>\n",
       "      <td>5</td>\n",
       "      <td>3.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-04</th>\n",
       "      <td>-0.424823</td>\n",
       "      <td>1.512963</td>\n",
       "      <td>-1.044807</td>\n",
       "      <td>5</td>\n",
       "      <td>4.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-05</th>\n",
       "      <td>0.425621</td>\n",
       "      <td>-0.049688</td>\n",
       "      <td>-0.433479</td>\n",
       "      <td>5</td>\n",
       "      <td>5.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-06</th>\n",
       "      <td>1.357304</td>\n",
       "      <td>0.178053</td>\n",
       "      <td>-1.265542</td>\n",
       "      <td>5</td>\n",
       "      <td>6.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-07</th>\n",
       "      <td>0.031349</td>\n",
       "      <td>-0.749075</td>\n",
       "      <td>-0.018996</td>\n",
       "      <td>5</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-08</th>\n",
       "      <td>-0.882429</td>\n",
       "      <td>-0.817032</td>\n",
       "      <td>-1.165577</td>\n",
       "      <td>5</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-09</th>\n",
       "      <td>1.548553</td>\n",
       "      <td>0.545712</td>\n",
       "      <td>-0.744566</td>\n",
       "      <td>5</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-10</th>\n",
       "      <td>0.733131</td>\n",
       "      <td>0.771151</td>\n",
       "      <td>-0.131902</td>\n",
       "      <td>5</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                   a         b         c  d    f\n",
       "2021-01-01  0.000000  0.000000  0.652312  5  1.0\n",
       "2021-01-02 -0.601569  0.888478  0.875262  5  2.0\n",
       "2021-01-03  0.971676  0.167080 -1.057748  5  3.0\n",
       "2021-01-04 -0.424823  1.512963 -1.044807  5  4.0\n",
       "2021-01-05  0.425621 -0.049688 -0.433479  5  5.0\n",
       "2021-01-06  1.357304  0.178053 -1.265542  5  6.0\n",
       "2021-01-07  0.031349 -0.749075 -0.018996  5  NaN\n",
       "2021-01-08 -0.882429 -0.817032 -1.165577  5  NaN\n",
       "2021-01-09  1.548553  0.545712 -0.744566  5  NaN\n",
       "2021-01-10  0.733131  0.771151 -0.131902  5  NaN"
      ]
     },
     "execution_count": 41,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.loc[:,\"d\"] = np.array([5] * len(df))\n",
    "df"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "A where operation with setting.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
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
       "      <th>a</th>\n",
       "      <th>b</th>\n",
       "      <th>c</th>\n",
       "      <th>d</th>\n",
       "      <th>f</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>2021-01-01</th>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>-0.652312</td>\n",
       "      <td>-5</td>\n",
       "      <td>-1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-02</th>\n",
       "      <td>-0.601569</td>\n",
       "      <td>-0.888478</td>\n",
       "      <td>-0.875262</td>\n",
       "      <td>-5</td>\n",
       "      <td>-2.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-03</th>\n",
       "      <td>-0.971676</td>\n",
       "      <td>-0.167080</td>\n",
       "      <td>-1.057748</td>\n",
       "      <td>-5</td>\n",
       "      <td>-3.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-04</th>\n",
       "      <td>-0.424823</td>\n",
       "      <td>-1.512963</td>\n",
       "      <td>-1.044807</td>\n",
       "      <td>-5</td>\n",
       "      <td>-4.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-05</th>\n",
       "      <td>-0.425621</td>\n",
       "      <td>-0.049688</td>\n",
       "      <td>-0.433479</td>\n",
       "      <td>-5</td>\n",
       "      <td>-5.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-06</th>\n",
       "      <td>-1.357304</td>\n",
       "      <td>-0.178053</td>\n",
       "      <td>-1.265542</td>\n",
       "      <td>-5</td>\n",
       "      <td>-6.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-07</th>\n",
       "      <td>-0.031349</td>\n",
       "      <td>-0.749075</td>\n",
       "      <td>-0.018996</td>\n",
       "      <td>-5</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-08</th>\n",
       "      <td>-0.882429</td>\n",
       "      <td>-0.817032</td>\n",
       "      <td>-1.165577</td>\n",
       "      <td>-5</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-09</th>\n",
       "      <td>-1.548553</td>\n",
       "      <td>-0.545712</td>\n",
       "      <td>-0.744566</td>\n",
       "      <td>-5</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-10</th>\n",
       "      <td>-0.733131</td>\n",
       "      <td>-0.771151</td>\n",
       "      <td>-0.131902</td>\n",
       "      <td>-5</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                   a         b         c  d    f\n",
       "2021-01-01  0.000000  0.000000 -0.652312 -5 -1.0\n",
       "2021-01-02 -0.601569 -0.888478 -0.875262 -5 -2.0\n",
       "2021-01-03 -0.971676 -0.167080 -1.057748 -5 -3.0\n",
       "2021-01-04 -0.424823 -1.512963 -1.044807 -5 -4.0\n",
       "2021-01-05 -0.425621 -0.049688 -0.433479 -5 -5.0\n",
       "2021-01-06 -1.357304 -0.178053 -1.265542 -5 -6.0\n",
       "2021-01-07 -0.031349 -0.749075 -0.018996 -5  NaN\n",
       "2021-01-08 -0.882429 -0.817032 -1.165577 -5  NaN\n",
       "2021-01-09 -1.548553 -0.545712 -0.744566 -5  NaN\n",
       "2021-01-10 -0.733131 -0.771151 -0.131902 -5  NaN"
      ]
     },
     "execution_count": 43,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df3 = df.copy()\n",
    "df3[df3 > 0] = -df3\n",
    "df3"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Missing Data\n",
    "\n",
    "pandas primarily uses the value np.nanto represent missing data. It is by default not included in\n",
    "computations. \n",
    "\n",
    "Reindexing allows you to change/add/delete the index on a specified axis. This returns a copy of\n",
    "the data."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
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
       "      <th>a</th>\n",
       "      <th>b</th>\n",
       "      <th>c</th>\n",
       "      <th>d</th>\n",
       "      <th>f</th>\n",
       "      <th>e</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>2021-01-01</th>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.652312</td>\n",
       "      <td>5</td>\n",
       "      <td>1.0</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-02</th>\n",
       "      <td>-0.601569</td>\n",
       "      <td>0.888478</td>\n",
       "      <td>0.875262</td>\n",
       "      <td>5</td>\n",
       "      <td>2.0</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-03</th>\n",
       "      <td>0.971676</td>\n",
       "      <td>0.167080</td>\n",
       "      <td>-1.057748</td>\n",
       "      <td>5</td>\n",
       "      <td>3.0</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-04</th>\n",
       "      <td>-0.424823</td>\n",
       "      <td>1.512963</td>\n",
       "      <td>-1.044807</td>\n",
       "      <td>5</td>\n",
       "      <td>4.0</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                   a         b         c  d    f   e\n",
       "2021-01-01  0.000000  0.000000  0.652312  5  1.0 NaN\n",
       "2021-01-02 -0.601569  0.888478  0.875262  5  2.0 NaN\n",
       "2021-01-03  0.971676  0.167080 -1.057748  5  3.0 NaN\n",
       "2021-01-04 -0.424823  1.512963 -1.044807  5  4.0 NaN"
      ]
     },
     "execution_count": 50,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df1 = df.reindex(index=dates[0:4], columns=list(df.columns) + [\"e\"])\n",
    "df1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "metadata": {},
   "outputs": [],
   "source": [
    "df1.loc[dates[0]:dates[1], \"e\"] = 1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
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
       "      <th>a</th>\n",
       "      <th>b</th>\n",
       "      <th>c</th>\n",
       "      <th>d</th>\n",
       "      <th>f</th>\n",
       "      <th>e</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>2021-01-01</th>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.652312</td>\n",
       "      <td>5</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-02</th>\n",
       "      <td>-0.601569</td>\n",
       "      <td>0.888478</td>\n",
       "      <td>0.875262</td>\n",
       "      <td>5</td>\n",
       "      <td>2.0</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-03</th>\n",
       "      <td>0.971676</td>\n",
       "      <td>0.167080</td>\n",
       "      <td>-1.057748</td>\n",
       "      <td>5</td>\n",
       "      <td>3.0</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-04</th>\n",
       "      <td>-0.424823</td>\n",
       "      <td>1.512963</td>\n",
       "      <td>-1.044807</td>\n",
       "      <td>5</td>\n",
       "      <td>4.0</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                   a         b         c  d    f    e\n",
       "2021-01-01  0.000000  0.000000  0.652312  5  1.0  1.0\n",
       "2021-01-02 -0.601569  0.888478  0.875262  5  2.0  1.0\n",
       "2021-01-03  0.971676  0.167080 -1.057748  5  3.0  NaN\n",
       "2021-01-04 -0.424823  1.512963 -1.044807  5  4.0  NaN"
      ]
     },
     "execution_count": 52,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df1"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "To drop any rows that have missing data."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
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
       "      <th>a</th>\n",
       "      <th>b</th>\n",
       "      <th>c</th>\n",
       "      <th>d</th>\n",
       "      <th>f</th>\n",
       "      <th>e</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>2021-01-01</th>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.652312</td>\n",
       "      <td>5</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-02</th>\n",
       "      <td>-0.601569</td>\n",
       "      <td>0.888478</td>\n",
       "      <td>0.875262</td>\n",
       "      <td>5</td>\n",
       "      <td>2.0</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                   a         b         c  d    f    e\n",
       "2021-01-01  0.000000  0.000000  0.652312  5  1.0  1.0\n",
       "2021-01-02 -0.601569  0.888478  0.875262  5  2.0  1.0"
      ]
     },
     "execution_count": 53,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df1.dropna(how=\"any\", inplace=True)\n",
    "df1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
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
       "      <th>a</th>\n",
       "      <th>b</th>\n",
       "      <th>c</th>\n",
       "      <th>d</th>\n",
       "      <th>f</th>\n",
       "      <th>e</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>2021-01-01</th>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.652312</td>\n",
       "      <td>5</td>\n",
       "      <td>1.0</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-02</th>\n",
       "      <td>-0.601569</td>\n",
       "      <td>0.888478</td>\n",
       "      <td>0.875262</td>\n",
       "      <td>5</td>\n",
       "      <td>2.0</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-03</th>\n",
       "      <td>0.971676</td>\n",
       "      <td>0.167080</td>\n",
       "      <td>-1.057748</td>\n",
       "      <td>5</td>\n",
       "      <td>3.0</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-04</th>\n",
       "      <td>-0.424823</td>\n",
       "      <td>1.512963</td>\n",
       "      <td>-1.044807</td>\n",
       "      <td>5</td>\n",
       "      <td>4.0</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                   a         b         c  d    f   e\n",
       "2021-01-01  0.000000  0.000000  0.652312  5  1.0 NaN\n",
       "2021-01-02 -0.601569  0.888478  0.875262  5  2.0 NaN\n",
       "2021-01-03  0.971676  0.167080 -1.057748  5  3.0 NaN\n",
       "2021-01-04 -0.424823  1.512963 -1.044807  5  4.0 NaN"
      ]
     },
     "execution_count": 54,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df1 = df.reindex(index=dates[0:4], columns=list(df.columns) + [\"e\"])\n",
    "df1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
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
       "      <th>a</th>\n",
       "      <th>b</th>\n",
       "      <th>c</th>\n",
       "      <th>d</th>\n",
       "      <th>f</th>\n",
       "      <th>e</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>2021-01-01</th>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.652312</td>\n",
       "      <td>5</td>\n",
       "      <td>1.0</td>\n",
       "      <td>4.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-02</th>\n",
       "      <td>-0.601569</td>\n",
       "      <td>0.888478</td>\n",
       "      <td>0.875262</td>\n",
       "      <td>5</td>\n",
       "      <td>2.0</td>\n",
       "      <td>4.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-03</th>\n",
       "      <td>0.971676</td>\n",
       "      <td>0.167080</td>\n",
       "      <td>-1.057748</td>\n",
       "      <td>5</td>\n",
       "      <td>3.0</td>\n",
       "      <td>4.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-04</th>\n",
       "      <td>-0.424823</td>\n",
       "      <td>1.512963</td>\n",
       "      <td>-1.044807</td>\n",
       "      <td>5</td>\n",
       "      <td>4.0</td>\n",
       "      <td>4.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                   a         b         c  d    f    e\n",
       "2021-01-01  0.000000  0.000000  0.652312  5  1.0  4.0\n",
       "2021-01-02 -0.601569  0.888478  0.875262  5  2.0  4.0\n",
       "2021-01-03  0.971676  0.167080 -1.057748  5  3.0  4.0\n",
       "2021-01-04 -0.424823  1.512963 -1.044807  5  4.0  4.0"
      ]
     },
     "execution_count": 57,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df1.fillna(value=4, inplace=True)\n",
    "df1"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "To get the boolean mask where values are nan"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
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
       "      <th>a</th>\n",
       "      <th>b</th>\n",
       "      <th>c</th>\n",
       "      <th>d</th>\n",
       "      <th>f</th>\n",
       "      <th>e</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>2021-01-01</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-02</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-03</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-04</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                a      b      c      d      f      e\n",
       "2021-01-01  False  False  False  False  False  False\n",
       "2021-01-02  False  False  False  False  False  False\n",
       "2021-01-03  False  False  False  False  False  False\n",
       "2021-01-04  False  False  False  False  False  False"
      ]
     },
     "execution_count": 58,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "pd.isnull(df1)"
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
