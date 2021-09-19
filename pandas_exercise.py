{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 38,
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
   "execution_count": 39,
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
     "execution_count": 39,
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
   "execution_count": 40,
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
     "execution_count": 40,
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
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>2021-01-01</th>\n",
       "      <td>0.480883</td>\n",
       "      <td>-0.168973</td>\n",
       "      <td>1.495480</td>\n",
       "      <td>-0.738783</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-02</th>\n",
       "      <td>-0.731000</td>\n",
       "      <td>1.349302</td>\n",
       "      <td>1.059063</td>\n",
       "      <td>0.710393</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-03</th>\n",
       "      <td>-1.902861</td>\n",
       "      <td>0.337020</td>\n",
       "      <td>-0.344063</td>\n",
       "      <td>0.099660</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-04</th>\n",
       "      <td>0.769069</td>\n",
       "      <td>-1.190182</td>\n",
       "      <td>-0.289454</td>\n",
       "      <td>-0.147524</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-05</th>\n",
       "      <td>1.318712</td>\n",
       "      <td>0.509324</td>\n",
       "      <td>0.663939</td>\n",
       "      <td>-0.471609</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-06</th>\n",
       "      <td>0.501530</td>\n",
       "      <td>-0.585483</td>\n",
       "      <td>-1.450538</td>\n",
       "      <td>1.048853</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-07</th>\n",
       "      <td>-2.224210</td>\n",
       "      <td>-1.439403</td>\n",
       "      <td>-1.494122</td>\n",
       "      <td>0.974578</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-08</th>\n",
       "      <td>-0.761851</td>\n",
       "      <td>-0.448527</td>\n",
       "      <td>-0.440605</td>\n",
       "      <td>-0.297513</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-09</th>\n",
       "      <td>0.783171</td>\n",
       "      <td>0.270960</td>\n",
       "      <td>-1.189345</td>\n",
       "      <td>-2.237883</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-10</th>\n",
       "      <td>0.210608</td>\n",
       "      <td>1.971006</td>\n",
       "      <td>-0.692841</td>\n",
       "      <td>0.344552</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                   a         b         c         d\n",
       "2021-01-01  0.480883 -0.168973  1.495480 -0.738783\n",
       "2021-01-02 -0.731000  1.349302  1.059063  0.710393\n",
       "2021-01-03 -1.902861  0.337020 -0.344063  0.099660\n",
       "2021-01-04  0.769069 -1.190182 -0.289454 -0.147524\n",
       "2021-01-05  1.318712  0.509324  0.663939 -0.471609\n",
       "2021-01-06  0.501530 -0.585483 -1.450538  1.048853\n",
       "2021-01-07 -2.224210 -1.439403 -1.494122  0.974578\n",
       "2021-01-08 -0.761851 -0.448527 -0.440605 -0.297513\n",
       "2021-01-09  0.783171  0.270960 -1.189345 -2.237883\n",
       "2021-01-10  0.210608  1.971006 -0.692841  0.344552"
      ]
     },
     "execution_count": 41,
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
   "execution_count": 42,
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
     "execution_count": 42,
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
   "execution_count": 43,
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
     "execution_count": 43,
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
   "execution_count": 44,
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
     "execution_count": 44,
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
   "execution_count": 45,
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
     "execution_count": 45,
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
   "execution_count": 46,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['a', 'b', 'c', 'd'], dtype='object')"
      ]
     },
     "execution_count": 46,
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
   "execution_count": 47,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ 0.48088295, -0.16897317,  1.49547953, -0.73878348],\n",
       "       [-0.73099987,  1.3493024 ,  1.05906305,  0.71039304],\n",
       "       [-1.90286104,  0.33701962, -0.34406281,  0.09966005],\n",
       "       [ 0.76906861, -1.19018158, -0.28945403, -0.14752402],\n",
       "       [ 1.318712  ,  0.50932405,  0.66393919, -0.47160912],\n",
       "       [ 0.50152981, -0.58548257, -1.45053834,  1.04885328],\n",
       "       [-2.22421041, -1.43940346, -1.49412225,  0.97457768],\n",
       "       [-0.76185132, -0.44852658, -0.44060479, -0.29751317],\n",
       "       [ 0.78317143,  0.27096016, -1.1893455 , -2.23788252],\n",
       "       [ 0.21060819,  1.97100559, -0.69284132,  0.34455183]])"
      ]
     },
     "execution_count": 47,
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
   "execution_count": 48,
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
       "      <td>-0.155595</td>\n",
       "      <td>0.060504</td>\n",
       "      <td>-0.268249</td>\n",
       "      <td>-0.071528</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>1.197759</td>\n",
       "      <td>1.063339</td>\n",
       "      <td>1.038516</td>\n",
       "      <td>0.972891</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>-2.224210</td>\n",
       "      <td>-1.439403</td>\n",
       "      <td>-1.494122</td>\n",
       "      <td>-2.237883</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>-0.754138</td>\n",
       "      <td>-0.551244</td>\n",
       "      <td>-1.065219</td>\n",
       "      <td>-0.428085</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>0.345746</td>\n",
       "      <td>0.050993</td>\n",
       "      <td>-0.392334</td>\n",
       "      <td>-0.023932</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>0.702184</td>\n",
       "      <td>0.466248</td>\n",
       "      <td>0.425591</td>\n",
       "      <td>0.618933</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>1.318712</td>\n",
       "      <td>1.971006</td>\n",
       "      <td>1.495480</td>\n",
       "      <td>1.048853</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "               a          b          c          d\n",
       "count  10.000000  10.000000  10.000000  10.000000\n",
       "mean   -0.155595   0.060504  -0.268249  -0.071528\n",
       "std     1.197759   1.063339   1.038516   0.972891\n",
       "min    -2.224210  -1.439403  -1.494122  -2.237883\n",
       "25%    -0.754138  -0.551244  -1.065219  -0.428085\n",
       "50%     0.345746   0.050993  -0.392334  -0.023932\n",
       "75%     0.702184   0.466248   0.425591   0.618933\n",
       "max     1.318712   1.971006   1.495480   1.048853"
      ]
     },
     "execution_count": 48,
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
   "execution_count": 49,
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
       "      <td>0.480883</td>\n",
       "      <td>-0.731000</td>\n",
       "      <td>-1.902861</td>\n",
       "      <td>0.769069</td>\n",
       "      <td>1.318712</td>\n",
       "      <td>0.501530</td>\n",
       "      <td>-2.224210</td>\n",
       "      <td>-0.761851</td>\n",
       "      <td>0.783171</td>\n",
       "      <td>0.210608</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>b</th>\n",
       "      <td>-0.168973</td>\n",
       "      <td>1.349302</td>\n",
       "      <td>0.337020</td>\n",
       "      <td>-1.190182</td>\n",
       "      <td>0.509324</td>\n",
       "      <td>-0.585483</td>\n",
       "      <td>-1.439403</td>\n",
       "      <td>-0.448527</td>\n",
       "      <td>0.270960</td>\n",
       "      <td>1.971006</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>c</th>\n",
       "      <td>1.495480</td>\n",
       "      <td>1.059063</td>\n",
       "      <td>-0.344063</td>\n",
       "      <td>-0.289454</td>\n",
       "      <td>0.663939</td>\n",
       "      <td>-1.450538</td>\n",
       "      <td>-1.494122</td>\n",
       "      <td>-0.440605</td>\n",
       "      <td>-1.189345</td>\n",
       "      <td>-0.692841</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>d</th>\n",
       "      <td>-0.738783</td>\n",
       "      <td>0.710393</td>\n",
       "      <td>0.099660</td>\n",
       "      <td>-0.147524</td>\n",
       "      <td>-0.471609</td>\n",
       "      <td>1.048853</td>\n",
       "      <td>0.974578</td>\n",
       "      <td>-0.297513</td>\n",
       "      <td>-2.237883</td>\n",
       "      <td>0.344552</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   2021-01-01  2021-01-02  2021-01-03  2021-01-04  2021-01-05  2021-01-06  \\\n",
       "a    0.480883   -0.731000   -1.902861    0.769069    1.318712    0.501530   \n",
       "b   -0.168973    1.349302    0.337020   -1.190182    0.509324   -0.585483   \n",
       "c    1.495480    1.059063   -0.344063   -0.289454    0.663939   -1.450538   \n",
       "d   -0.738783    0.710393    0.099660   -0.147524   -0.471609    1.048853   \n",
       "\n",
       "   2021-01-07  2021-01-08  2021-01-09  2021-01-10  \n",
       "a   -2.224210   -0.761851    0.783171    0.210608  \n",
       "b   -1.439403   -0.448527    0.270960    1.971006  \n",
       "c   -1.494122   -0.440605   -1.189345   -0.692841  \n",
       "d    0.974578   -0.297513   -2.237883    0.344552  "
      ]
     },
     "execution_count": 49,
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
   "execution_count": 51,
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
       "      <td>0.480883</td>\n",
       "      <td>-0.168973</td>\n",
       "      <td>1.495480</td>\n",
       "      <td>-0.738783</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-02</th>\n",
       "      <td>-0.731000</td>\n",
       "      <td>1.349302</td>\n",
       "      <td>1.059063</td>\n",
       "      <td>0.710393</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-03</th>\n",
       "      <td>-1.902861</td>\n",
       "      <td>0.337020</td>\n",
       "      <td>-0.344063</td>\n",
       "      <td>0.099660</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-04</th>\n",
       "      <td>0.769069</td>\n",
       "      <td>-1.190182</td>\n",
       "      <td>-0.289454</td>\n",
       "      <td>-0.147524</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-05</th>\n",
       "      <td>1.318712</td>\n",
       "      <td>0.509324</td>\n",
       "      <td>0.663939</td>\n",
       "      <td>-0.471609</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-06</th>\n",
       "      <td>0.501530</td>\n",
       "      <td>-0.585483</td>\n",
       "      <td>-1.450538</td>\n",
       "      <td>1.048853</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-07</th>\n",
       "      <td>-2.224210</td>\n",
       "      <td>-1.439403</td>\n",
       "      <td>-1.494122</td>\n",
       "      <td>0.974578</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-08</th>\n",
       "      <td>-0.761851</td>\n",
       "      <td>-0.448527</td>\n",
       "      <td>-0.440605</td>\n",
       "      <td>-0.297513</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-09</th>\n",
       "      <td>0.783171</td>\n",
       "      <td>0.270960</td>\n",
       "      <td>-1.189345</td>\n",
       "      <td>-2.237883</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-10</th>\n",
       "      <td>0.210608</td>\n",
       "      <td>1.971006</td>\n",
       "      <td>-0.692841</td>\n",
       "      <td>0.344552</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                   a         b         c         d\n",
       "2021-01-01  0.480883 -0.168973  1.495480 -0.738783\n",
       "2021-01-02 -0.731000  1.349302  1.059063  0.710393\n",
       "2021-01-03 -1.902861  0.337020 -0.344063  0.099660\n",
       "2021-01-04  0.769069 -1.190182 -0.289454 -0.147524\n",
       "2021-01-05  1.318712  0.509324  0.663939 -0.471609\n",
       "2021-01-06  0.501530 -0.585483 -1.450538  1.048853\n",
       "2021-01-07 -2.224210 -1.439403 -1.494122  0.974578\n",
       "2021-01-08 -0.761851 -0.448527 -0.440605 -0.297513\n",
       "2021-01-09  0.783171  0.270960 -1.189345 -2.237883\n",
       "2021-01-10  0.210608  1.971006 -0.692841  0.344552"
      ]
     },
     "execution_count": 51,
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
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>2021-01-07</th>\n",
       "      <td>-2.224210</td>\n",
       "      <td>-1.439403</td>\n",
       "      <td>-1.494122</td>\n",
       "      <td>0.974578</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-06</th>\n",
       "      <td>0.501530</td>\n",
       "      <td>-0.585483</td>\n",
       "      <td>-1.450538</td>\n",
       "      <td>1.048853</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-09</th>\n",
       "      <td>0.783171</td>\n",
       "      <td>0.270960</td>\n",
       "      <td>-1.189345</td>\n",
       "      <td>-2.237883</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-10</th>\n",
       "      <td>0.210608</td>\n",
       "      <td>1.971006</td>\n",
       "      <td>-0.692841</td>\n",
       "      <td>0.344552</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-08</th>\n",
       "      <td>-0.761851</td>\n",
       "      <td>-0.448527</td>\n",
       "      <td>-0.440605</td>\n",
       "      <td>-0.297513</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-03</th>\n",
       "      <td>-1.902861</td>\n",
       "      <td>0.337020</td>\n",
       "      <td>-0.344063</td>\n",
       "      <td>0.099660</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-04</th>\n",
       "      <td>0.769069</td>\n",
       "      <td>-1.190182</td>\n",
       "      <td>-0.289454</td>\n",
       "      <td>-0.147524</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-05</th>\n",
       "      <td>1.318712</td>\n",
       "      <td>0.509324</td>\n",
       "      <td>0.663939</td>\n",
       "      <td>-0.471609</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-02</th>\n",
       "      <td>-0.731000</td>\n",
       "      <td>1.349302</td>\n",
       "      <td>1.059063</td>\n",
       "      <td>0.710393</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-01</th>\n",
       "      <td>0.480883</td>\n",
       "      <td>-0.168973</td>\n",
       "      <td>1.495480</td>\n",
       "      <td>-0.738783</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                   a         b         c         d\n",
       "2021-01-07 -2.224210 -1.439403 -1.494122  0.974578\n",
       "2021-01-06  0.501530 -0.585483 -1.450538  1.048853\n",
       "2021-01-09  0.783171  0.270960 -1.189345 -2.237883\n",
       "2021-01-10  0.210608  1.971006 -0.692841  0.344552\n",
       "2021-01-08 -0.761851 -0.448527 -0.440605 -0.297513\n",
       "2021-01-03 -1.902861  0.337020 -0.344063  0.099660\n",
       "2021-01-04  0.769069 -1.190182 -0.289454 -0.147524\n",
       "2021-01-05  1.318712  0.509324  0.663939 -0.471609\n",
       "2021-01-02 -0.731000  1.349302  1.059063  0.710393\n",
       "2021-01-01  0.480883 -0.168973  1.495480 -0.738783"
      ]
     },
     "execution_count": 52,
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
   "execution_count": 53,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "2021-01-01    0.480883\n",
       "2021-01-02   -0.731000\n",
       "2021-01-03   -1.902861\n",
       "2021-01-04    0.769069\n",
       "2021-01-05    1.318712\n",
       "2021-01-06    0.501530\n",
       "2021-01-07   -2.224210\n",
       "2021-01-08   -0.761851\n",
       "2021-01-09    0.783171\n",
       "2021-01-10    0.210608\n",
       "Freq: D, Name: a, dtype: float64"
      ]
     },
     "execution_count": 53,
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
   "execution_count": 54,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "2021-01-01    0.480883\n",
       "2021-01-02   -0.731000\n",
       "2021-01-03   -1.902861\n",
       "2021-01-04    0.769069\n",
       "2021-01-05    1.318712\n",
       "2021-01-06    0.501530\n",
       "2021-01-07   -2.224210\n",
       "2021-01-08   -0.761851\n",
       "2021-01-09    0.783171\n",
       "2021-01-10    0.210608\n",
       "Freq: D, Name: a, dtype: float64"
      ]
     },
     "execution_count": 54,
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
   "execution_count": 55,
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
       "      <td>0.480883</td>\n",
       "      <td>-0.168973</td>\n",
       "      <td>1.495480</td>\n",
       "      <td>-0.738783</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-02</th>\n",
       "      <td>-0.731000</td>\n",
       "      <td>1.349302</td>\n",
       "      <td>1.059063</td>\n",
       "      <td>0.710393</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-03</th>\n",
       "      <td>-1.902861</td>\n",
       "      <td>0.337020</td>\n",
       "      <td>-0.344063</td>\n",
       "      <td>0.099660</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-04</th>\n",
       "      <td>0.769069</td>\n",
       "      <td>-1.190182</td>\n",
       "      <td>-0.289454</td>\n",
       "      <td>-0.147524</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                   a         b         c         d\n",
       "2021-01-01  0.480883 -0.168973  1.495480 -0.738783\n",
       "2021-01-02 -0.731000  1.349302  1.059063  0.710393\n",
       "2021-01-03 -1.902861  0.337020 -0.344063  0.099660\n",
       "2021-01-04  0.769069 -1.190182 -0.289454 -0.147524"
      ]
     },
     "execution_count": 55,
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
   "execution_count": 40,
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
       "      <td>-1.047581</td>\n",
       "      <td>1.161637</td>\n",
       "      <td>-0.803006</td>\n",
       "      <td>-2.093864</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-03</th>\n",
       "      <td>-1.151396</td>\n",
       "      <td>0.575674</td>\n",
       "      <td>-0.723401</td>\n",
       "      <td>-0.487148</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-04</th>\n",
       "      <td>-0.628977</td>\n",
       "      <td>1.176879</td>\n",
       "      <td>-0.412652</td>\n",
       "      <td>2.204962</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-05</th>\n",
       "      <td>1.115872</td>\n",
       "      <td>1.940716</td>\n",
       "      <td>0.584026</td>\n",
       "      <td>0.259215</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-06</th>\n",
       "      <td>0.944919</td>\n",
       "      <td>-1.035663</td>\n",
       "      <td>-0.510705</td>\n",
       "      <td>-1.078126</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-07</th>\n",
       "      <td>-0.581449</td>\n",
       "      <td>1.246238</td>\n",
       "      <td>-0.510814</td>\n",
       "      <td>-1.517736</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                   a         b         c         d\n",
       "2021-01-02 -1.047581  1.161637 -0.803006 -2.093864\n",
       "2021-01-03 -1.151396  0.575674 -0.723401 -0.487148\n",
       "2021-01-04 -0.628977  1.176879 -0.412652  2.204962\n",
       "2021-01-05  1.115872  1.940716  0.584026  0.259215\n",
       "2021-01-06  0.944919 -1.035663 -0.510705 -1.078126\n",
       "2021-01-07 -0.581449  1.246238 -0.510814 -1.517736"
      ]
     },
     "execution_count": 40,
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
   "execution_count": 56,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "a    0.480883\n",
       "b   -0.168973\n",
       "c    1.495480\n",
       "d   -0.738783\n",
       "Name: 2021-01-01 00:00:00, dtype: float64"
      ]
     },
     "execution_count": 56,
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
       "      <th>b</th>\n",
       "      <th>c</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>2021-01-01</th>\n",
       "      <td>-0.168973</td>\n",
       "      <td>1.495480</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-02</th>\n",
       "      <td>1.349302</td>\n",
       "      <td>1.059063</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-03</th>\n",
       "      <td>0.337020</td>\n",
       "      <td>-0.344063</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-04</th>\n",
       "      <td>-1.190182</td>\n",
       "      <td>-0.289454</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-05</th>\n",
       "      <td>0.509324</td>\n",
       "      <td>0.663939</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-06</th>\n",
       "      <td>-0.585483</td>\n",
       "      <td>-1.450538</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-07</th>\n",
       "      <td>-1.439403</td>\n",
       "      <td>-1.494122</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-08</th>\n",
       "      <td>-0.448527</td>\n",
       "      <td>-0.440605</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-09</th>\n",
       "      <td>0.270960</td>\n",
       "      <td>-1.189345</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-10</th>\n",
       "      <td>1.971006</td>\n",
       "      <td>-0.692841</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                   b         c\n",
       "2021-01-01 -0.168973  1.495480\n",
       "2021-01-02  1.349302  1.059063\n",
       "2021-01-03  0.337020 -0.344063\n",
       "2021-01-04 -1.190182 -0.289454\n",
       "2021-01-05  0.509324  0.663939\n",
       "2021-01-06 -0.585483 -1.450538\n",
       "2021-01-07 -1.439403 -1.494122\n",
       "2021-01-08 -0.448527 -0.440605\n",
       "2021-01-09  0.270960 -1.189345\n",
       "2021-01-10  1.971006 -0.692841"
      ]
     },
     "execution_count": 57,
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
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>2021-01-03</th>\n",
       "      <td>-1.151396</td>\n",
       "      <td>0.575674</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-04</th>\n",
       "      <td>-0.628977</td>\n",
       "      <td>1.176879</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-05</th>\n",
       "      <td>1.115872</td>\n",
       "      <td>1.940716</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-06</th>\n",
       "      <td>0.944919</td>\n",
       "      <td>-1.035663</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                   a         b\n",
       "2021-01-03 -1.151396  0.575674\n",
       "2021-01-04 -0.628977  1.176879\n",
       "2021-01-05  1.115872  1.940716\n",
       "2021-01-06  0.944919 -1.035663"
      ]
     },
     "execution_count": 43,
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
   "execution_count": 58,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "c   -0.344063\n",
       "d    0.099660\n",
       "Name: 2021-01-03 00:00:00, dtype: float64"
      ]
     },
     "execution_count": 58,
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
   "execution_count": 45,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "-1.0475811442402923"
      ]
     },
     "execution_count": 45,
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
   "execution_count": 47,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "-1.0475811442402923"
      ]
     },
     "execution_count": 47,
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
   "execution_count": 61,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "a    0.769069\n",
       "b   -1.190182\n",
       "c   -0.289454\n",
       "d   -0.147524\n",
       "Name: 2021-01-04 00:00:00, dtype: float64"
      ]
     },
     "execution_count": 61,
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
   "execution_count": 60,
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
       "      <td>-0.289454</td>\n",
       "      <td>-0.147524</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-05</th>\n",
       "      <td>0.663939</td>\n",
       "      <td>-0.471609</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                   c         d\n",
       "2021-01-04 -0.289454 -0.147524\n",
       "2021-01-05  0.663939 -0.471609"
      ]
     },
     "execution_count": 60,
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
   "execution_count": 59,
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
       "      <td>-0.731000</td>\n",
       "      <td>1.059063</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-03</th>\n",
       "      <td>-1.902861</td>\n",
       "      <td>-0.344063</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-05</th>\n",
       "      <td>1.318712</td>\n",
       "      <td>0.663939</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                   a         c\n",
       "2021-01-02 -0.731000  1.059063\n",
       "2021-01-03 -1.902861 -0.344063\n",
       "2021-01-05  1.318712  0.663939"
      ]
     },
     "execution_count": 59,
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
   "execution_count": 63,
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
       "      <td>-0.731000</td>\n",
       "      <td>1.349302</td>\n",
       "      <td>1.059063</td>\n",
       "      <td>0.710393</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-03</th>\n",
       "      <td>-1.902861</td>\n",
       "      <td>0.337020</td>\n",
       "      <td>-0.344063</td>\n",
       "      <td>0.099660</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                   a         b         c         d\n",
       "2021-01-02 -0.731000  1.349302  1.059063  0.710393\n",
       "2021-01-03 -1.902861  0.337020 -0.344063  0.099660"
      ]
     },
     "execution_count": 63,
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
   "execution_count": 62,
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
       "      <td>-0.168973</td>\n",
       "      <td>1.495480</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-02</th>\n",
       "      <td>1.349302</td>\n",
       "      <td>1.059063</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-03</th>\n",
       "      <td>0.337020</td>\n",
       "      <td>-0.344063</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-04</th>\n",
       "      <td>-1.190182</td>\n",
       "      <td>-0.289454</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-05</th>\n",
       "      <td>0.509324</td>\n",
       "      <td>0.663939</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-06</th>\n",
       "      <td>-0.585483</td>\n",
       "      <td>-1.450538</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-07</th>\n",
       "      <td>-1.439403</td>\n",
       "      <td>-1.494122</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-08</th>\n",
       "      <td>-0.448527</td>\n",
       "      <td>-0.440605</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-09</th>\n",
       "      <td>0.270960</td>\n",
       "      <td>-1.189345</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-10</th>\n",
       "      <td>1.971006</td>\n",
       "      <td>-0.692841</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                   b         c\n",
       "2021-01-01 -0.168973  1.495480\n",
       "2021-01-02  1.349302  1.059063\n",
       "2021-01-03  0.337020 -0.344063\n",
       "2021-01-04 -1.190182 -0.289454\n",
       "2021-01-05  0.509324  0.663939\n",
       "2021-01-06 -0.585483 -1.450538\n",
       "2021-01-07 -1.439403 -1.494122\n",
       "2021-01-08 -0.448527 -0.440605\n",
       "2021-01-09  0.270960 -1.189345\n",
       "2021-01-10  1.971006 -0.692841"
      ]
     },
     "execution_count": 62,
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
   "execution_count": 65,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1.3493024012593835"
      ]
     },
     "execution_count": 65,
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
   "execution_count": 66,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1.3493024012593835"
      ]
     },
     "execution_count": 66,
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
   "execution_count": 67,
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
       "      <td>0.480883</td>\n",
       "      <td>-0.168973</td>\n",
       "      <td>1.495480</td>\n",
       "      <td>-0.738783</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-04</th>\n",
       "      <td>0.769069</td>\n",
       "      <td>-1.190182</td>\n",
       "      <td>-0.289454</td>\n",
       "      <td>-0.147524</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-05</th>\n",
       "      <td>1.318712</td>\n",
       "      <td>0.509324</td>\n",
       "      <td>0.663939</td>\n",
       "      <td>-0.471609</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-06</th>\n",
       "      <td>0.501530</td>\n",
       "      <td>-0.585483</td>\n",
       "      <td>-1.450538</td>\n",
       "      <td>1.048853</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-09</th>\n",
       "      <td>0.783171</td>\n",
       "      <td>0.270960</td>\n",
       "      <td>-1.189345</td>\n",
       "      <td>-2.237883</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-10</th>\n",
       "      <td>0.210608</td>\n",
       "      <td>1.971006</td>\n",
       "      <td>-0.692841</td>\n",
       "      <td>0.344552</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                   a         b         c         d\n",
       "2021-01-01  0.480883 -0.168973  1.495480 -0.738783\n",
       "2021-01-04  0.769069 -1.190182 -0.289454 -0.147524\n",
       "2021-01-05  1.318712  0.509324  0.663939 -0.471609\n",
       "2021-01-06  0.501530 -0.585483 -1.450538  1.048853\n",
       "2021-01-09  0.783171  0.270960 -1.189345 -2.237883\n",
       "2021-01-10  0.210608  1.971006 -0.692841  0.344552"
      ]
     },
     "execution_count": 67,
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
   "execution_count": 68,
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
       "      <td>0.480883</td>\n",
       "      <td>NaN</td>\n",
       "      <td>1.495480</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-02</th>\n",
       "      <td>NaN</td>\n",
       "      <td>1.349302</td>\n",
       "      <td>1.059063</td>\n",
       "      <td>0.710393</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-03</th>\n",
       "      <td>NaN</td>\n",
       "      <td>0.337020</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0.099660</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-04</th>\n",
       "      <td>0.769069</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-05</th>\n",
       "      <td>1.318712</td>\n",
       "      <td>0.509324</td>\n",
       "      <td>0.663939</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-06</th>\n",
       "      <td>0.501530</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>1.048853</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-07</th>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0.974578</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-08</th>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-09</th>\n",
       "      <td>0.783171</td>\n",
       "      <td>0.270960</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-10</th>\n",
       "      <td>0.210608</td>\n",
       "      <td>1.971006</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0.344552</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                   a         b         c         d\n",
       "2021-01-01  0.480883       NaN  1.495480       NaN\n",
       "2021-01-02       NaN  1.349302  1.059063  0.710393\n",
       "2021-01-03       NaN  0.337020       NaN  0.099660\n",
       "2021-01-04  0.769069       NaN       NaN       NaN\n",
       "2021-01-05  1.318712  0.509324  0.663939       NaN\n",
       "2021-01-06  0.501530       NaN       NaN  1.048853\n",
       "2021-01-07       NaN       NaN       NaN  0.974578\n",
       "2021-01-08       NaN       NaN       NaN       NaN\n",
       "2021-01-09  0.783171  0.270960       NaN       NaN\n",
       "2021-01-10  0.210608  1.971006       NaN  0.344552"
      ]
     },
     "execution_count": 68,
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
   "execution_count": 69,
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
       "      <td>0.480883</td>\n",
       "      <td>-0.168973</td>\n",
       "      <td>1.495480</td>\n",
       "      <td>-0.738783</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-02</th>\n",
       "      <td>-0.731000</td>\n",
       "      <td>1.349302</td>\n",
       "      <td>1.059063</td>\n",
       "      <td>0.710393</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-03</th>\n",
       "      <td>-1.902861</td>\n",
       "      <td>0.337020</td>\n",
       "      <td>-0.344063</td>\n",
       "      <td>0.099660</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-04</th>\n",
       "      <td>0.769069</td>\n",
       "      <td>-1.190182</td>\n",
       "      <td>-0.289454</td>\n",
       "      <td>-0.147524</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-05</th>\n",
       "      <td>1.318712</td>\n",
       "      <td>0.509324</td>\n",
       "      <td>0.663939</td>\n",
       "      <td>-0.471609</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-06</th>\n",
       "      <td>0.501530</td>\n",
       "      <td>-0.585483</td>\n",
       "      <td>-1.450538</td>\n",
       "      <td>1.048853</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-07</th>\n",
       "      <td>-2.224210</td>\n",
       "      <td>-1.439403</td>\n",
       "      <td>-1.494122</td>\n",
       "      <td>0.974578</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-08</th>\n",
       "      <td>-0.761851</td>\n",
       "      <td>-0.448527</td>\n",
       "      <td>-0.440605</td>\n",
       "      <td>-0.297513</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-09</th>\n",
       "      <td>0.783171</td>\n",
       "      <td>0.270960</td>\n",
       "      <td>-1.189345</td>\n",
       "      <td>-2.237883</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-10</th>\n",
       "      <td>0.210608</td>\n",
       "      <td>1.971006</td>\n",
       "      <td>-0.692841</td>\n",
       "      <td>0.344552</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                   a         b         c         d\n",
       "2021-01-01  0.480883 -0.168973  1.495480 -0.738783\n",
       "2021-01-02 -0.731000  1.349302  1.059063  0.710393\n",
       "2021-01-03 -1.902861  0.337020 -0.344063  0.099660\n",
       "2021-01-04  0.769069 -1.190182 -0.289454 -0.147524\n",
       "2021-01-05  1.318712  0.509324  0.663939 -0.471609\n",
       "2021-01-06  0.501530 -0.585483 -1.450538  1.048853\n",
       "2021-01-07 -2.224210 -1.439403 -1.494122  0.974578\n",
       "2021-01-08 -0.761851 -0.448527 -0.440605 -0.297513\n",
       "2021-01-09  0.783171  0.270960 -1.189345 -2.237883\n",
       "2021-01-10  0.210608  1.971006 -0.692841  0.344552"
      ]
     },
     "execution_count": 69,
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
   "execution_count": 70,
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
       "      <td>0.480883</td>\n",
       "      <td>-0.168973</td>\n",
       "      <td>1.495480</td>\n",
       "      <td>-0.738783</td>\n",
       "      <td>one</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-02</th>\n",
       "      <td>-0.731000</td>\n",
       "      <td>1.349302</td>\n",
       "      <td>1.059063</td>\n",
       "      <td>0.710393</td>\n",
       "      <td>two</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-03</th>\n",
       "      <td>-1.902861</td>\n",
       "      <td>0.337020</td>\n",
       "      <td>-0.344063</td>\n",
       "      <td>0.099660</td>\n",
       "      <td>three</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-04</th>\n",
       "      <td>0.769069</td>\n",
       "      <td>-1.190182</td>\n",
       "      <td>-0.289454</td>\n",
       "      <td>-0.147524</td>\n",
       "      <td>four</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-05</th>\n",
       "      <td>1.318712</td>\n",
       "      <td>0.509324</td>\n",
       "      <td>0.663939</td>\n",
       "      <td>-0.471609</td>\n",
       "      <td>five</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-06</th>\n",
       "      <td>0.501530</td>\n",
       "      <td>-0.585483</td>\n",
       "      <td>-1.450538</td>\n",
       "      <td>1.048853</td>\n",
       "      <td>one</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-07</th>\n",
       "      <td>-2.224210</td>\n",
       "      <td>-1.439403</td>\n",
       "      <td>-1.494122</td>\n",
       "      <td>0.974578</td>\n",
       "      <td>two</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-08</th>\n",
       "      <td>-0.761851</td>\n",
       "      <td>-0.448527</td>\n",
       "      <td>-0.440605</td>\n",
       "      <td>-0.297513</td>\n",
       "      <td>three</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-09</th>\n",
       "      <td>0.783171</td>\n",
       "      <td>0.270960</td>\n",
       "      <td>-1.189345</td>\n",
       "      <td>-2.237883</td>\n",
       "      <td>four</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-10</th>\n",
       "      <td>0.210608</td>\n",
       "      <td>1.971006</td>\n",
       "      <td>-0.692841</td>\n",
       "      <td>0.344552</td>\n",
       "      <td>five</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                   a         b         c         d      e\n",
       "2021-01-01  0.480883 -0.168973  1.495480 -0.738783    one\n",
       "2021-01-02 -0.731000  1.349302  1.059063  0.710393    two\n",
       "2021-01-03 -1.902861  0.337020 -0.344063  0.099660  three\n",
       "2021-01-04  0.769069 -1.190182 -0.289454 -0.147524   four\n",
       "2021-01-05  1.318712  0.509324  0.663939 -0.471609   five\n",
       "2021-01-06  0.501530 -0.585483 -1.450538  1.048853    one\n",
       "2021-01-07 -2.224210 -1.439403 -1.494122  0.974578    two\n",
       "2021-01-08 -0.761851 -0.448527 -0.440605 -0.297513  three\n",
       "2021-01-09  0.783171  0.270960 -1.189345 -2.237883   four\n",
       "2021-01-10  0.210608  1.971006 -0.692841  0.344552   five"
      ]
     },
     "execution_count": 70,
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
   "execution_count": 71,
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
       "      <td>0.480883</td>\n",
       "      <td>-0.168973</td>\n",
       "      <td>1.495480</td>\n",
       "      <td>-0.738783</td>\n",
       "      <td>one</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-02</th>\n",
       "      <td>-0.731000</td>\n",
       "      <td>1.349302</td>\n",
       "      <td>1.059063</td>\n",
       "      <td>0.710393</td>\n",
       "      <td>two</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-06</th>\n",
       "      <td>0.501530</td>\n",
       "      <td>-0.585483</td>\n",
       "      <td>-1.450538</td>\n",
       "      <td>1.048853</td>\n",
       "      <td>one</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-07</th>\n",
       "      <td>-2.224210</td>\n",
       "      <td>-1.439403</td>\n",
       "      <td>-1.494122</td>\n",
       "      <td>0.974578</td>\n",
       "      <td>two</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                   a         b         c         d    e\n",
       "2021-01-01  0.480883 -0.168973  1.495480 -0.738783  one\n",
       "2021-01-02 -0.731000  1.349302  1.059063  0.710393  two\n",
       "2021-01-06  0.501530 -0.585483 -1.450538  1.048853  one\n",
       "2021-01-07 -2.224210 -1.439403 -1.494122  0.974578  two"
      ]
     },
     "execution_count": 71,
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
   "execution_count": 72,
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
     "execution_count": 72,
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
   "execution_count": 73,
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
       "      <td>0.480883</td>\n",
       "      <td>-0.168973</td>\n",
       "      <td>1.495480</td>\n",
       "      <td>-0.738783</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-02</th>\n",
       "      <td>-0.731000</td>\n",
       "      <td>1.349302</td>\n",
       "      <td>1.059063</td>\n",
       "      <td>0.710393</td>\n",
       "      <td>2.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-03</th>\n",
       "      <td>-1.902861</td>\n",
       "      <td>0.337020</td>\n",
       "      <td>-0.344063</td>\n",
       "      <td>0.099660</td>\n",
       "      <td>3.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-04</th>\n",
       "      <td>0.769069</td>\n",
       "      <td>-1.190182</td>\n",
       "      <td>-0.289454</td>\n",
       "      <td>-0.147524</td>\n",
       "      <td>4.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-05</th>\n",
       "      <td>1.318712</td>\n",
       "      <td>0.509324</td>\n",
       "      <td>0.663939</td>\n",
       "      <td>-0.471609</td>\n",
       "      <td>5.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-06</th>\n",
       "      <td>0.501530</td>\n",
       "      <td>-0.585483</td>\n",
       "      <td>-1.450538</td>\n",
       "      <td>1.048853</td>\n",
       "      <td>6.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-07</th>\n",
       "      <td>-2.224210</td>\n",
       "      <td>-1.439403</td>\n",
       "      <td>-1.494122</td>\n",
       "      <td>0.974578</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-08</th>\n",
       "      <td>-0.761851</td>\n",
       "      <td>-0.448527</td>\n",
       "      <td>-0.440605</td>\n",
       "      <td>-0.297513</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-09</th>\n",
       "      <td>0.783171</td>\n",
       "      <td>0.270960</td>\n",
       "      <td>-1.189345</td>\n",
       "      <td>-2.237883</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-10</th>\n",
       "      <td>0.210608</td>\n",
       "      <td>1.971006</td>\n",
       "      <td>-0.692841</td>\n",
       "      <td>0.344552</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                   a         b         c         d    f\n",
       "2021-01-01  0.480883 -0.168973  1.495480 -0.738783  1.0\n",
       "2021-01-02 -0.731000  1.349302  1.059063  0.710393  2.0\n",
       "2021-01-03 -1.902861  0.337020 -0.344063  0.099660  3.0\n",
       "2021-01-04  0.769069 -1.190182 -0.289454 -0.147524  4.0\n",
       "2021-01-05  1.318712  0.509324  0.663939 -0.471609  5.0\n",
       "2021-01-06  0.501530 -0.585483 -1.450538  1.048853  6.0\n",
       "2021-01-07 -2.224210 -1.439403 -1.494122  0.974578  NaN\n",
       "2021-01-08 -0.761851 -0.448527 -0.440605 -0.297513  NaN\n",
       "2021-01-09  0.783171  0.270960 -1.189345 -2.237883  NaN\n",
       "2021-01-10  0.210608  1.971006 -0.692841  0.344552  NaN"
      ]
     },
     "execution_count": 73,
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
   "execution_count": 74,
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
   "execution_count": 75,
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
       "      <td>1.495480</td>\n",
       "      <td>5</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-02</th>\n",
       "      <td>-0.731000</td>\n",
       "      <td>1.349302</td>\n",
       "      <td>1.059063</td>\n",
       "      <td>5</td>\n",
       "      <td>2.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-03</th>\n",
       "      <td>-1.902861</td>\n",
       "      <td>0.337020</td>\n",
       "      <td>-0.344063</td>\n",
       "      <td>5</td>\n",
       "      <td>3.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-04</th>\n",
       "      <td>0.769069</td>\n",
       "      <td>-1.190182</td>\n",
       "      <td>-0.289454</td>\n",
       "      <td>5</td>\n",
       "      <td>4.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-05</th>\n",
       "      <td>1.318712</td>\n",
       "      <td>0.509324</td>\n",
       "      <td>0.663939</td>\n",
       "      <td>5</td>\n",
       "      <td>5.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-06</th>\n",
       "      <td>0.501530</td>\n",
       "      <td>-0.585483</td>\n",
       "      <td>-1.450538</td>\n",
       "      <td>5</td>\n",
       "      <td>6.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-07</th>\n",
       "      <td>-2.224210</td>\n",
       "      <td>-1.439403</td>\n",
       "      <td>-1.494122</td>\n",
       "      <td>5</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-08</th>\n",
       "      <td>-0.761851</td>\n",
       "      <td>-0.448527</td>\n",
       "      <td>-0.440605</td>\n",
       "      <td>5</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-09</th>\n",
       "      <td>0.783171</td>\n",
       "      <td>0.270960</td>\n",
       "      <td>-1.189345</td>\n",
       "      <td>5</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-10</th>\n",
       "      <td>0.210608</td>\n",
       "      <td>1.971006</td>\n",
       "      <td>-0.692841</td>\n",
       "      <td>5</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                   a         b         c  d    f\n",
       "2021-01-01  0.000000  0.000000  1.495480  5  1.0\n",
       "2021-01-02 -0.731000  1.349302  1.059063  5  2.0\n",
       "2021-01-03 -1.902861  0.337020 -0.344063  5  3.0\n",
       "2021-01-04  0.769069 -1.190182 -0.289454  5  4.0\n",
       "2021-01-05  1.318712  0.509324  0.663939  5  5.0\n",
       "2021-01-06  0.501530 -0.585483 -1.450538  5  6.0\n",
       "2021-01-07 -2.224210 -1.439403 -1.494122  5  NaN\n",
       "2021-01-08 -0.761851 -0.448527 -0.440605  5  NaN\n",
       "2021-01-09  0.783171  0.270960 -1.189345  5  NaN\n",
       "2021-01-10  0.210608  1.971006 -0.692841  5  NaN"
      ]
     },
     "execution_count": 76,
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
   "execution_count": 78,
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
       "      <td>-1.495480</td>\n",
       "      <td>-5</td>\n",
       "      <td>-1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-02</th>\n",
       "      <td>-0.731000</td>\n",
       "      <td>-1.349302</td>\n",
       "      <td>-1.059063</td>\n",
       "      <td>-5</td>\n",
       "      <td>-2.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-03</th>\n",
       "      <td>-1.902861</td>\n",
       "      <td>-0.337020</td>\n",
       "      <td>-0.344063</td>\n",
       "      <td>-5</td>\n",
       "      <td>-3.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-04</th>\n",
       "      <td>-0.769069</td>\n",
       "      <td>-1.190182</td>\n",
       "      <td>-0.289454</td>\n",
       "      <td>-5</td>\n",
       "      <td>-4.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-05</th>\n",
       "      <td>-1.318712</td>\n",
       "      <td>-0.509324</td>\n",
       "      <td>-0.663939</td>\n",
       "      <td>-5</td>\n",
       "      <td>-5.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-06</th>\n",
       "      <td>-0.501530</td>\n",
       "      <td>-0.585483</td>\n",
       "      <td>-1.450538</td>\n",
       "      <td>-5</td>\n",
       "      <td>-6.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-07</th>\n",
       "      <td>-2.224210</td>\n",
       "      <td>-1.439403</td>\n",
       "      <td>-1.494122</td>\n",
       "      <td>-5</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-08</th>\n",
       "      <td>-0.761851</td>\n",
       "      <td>-0.448527</td>\n",
       "      <td>-0.440605</td>\n",
       "      <td>-5</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-09</th>\n",
       "      <td>-0.783171</td>\n",
       "      <td>-0.270960</td>\n",
       "      <td>-1.189345</td>\n",
       "      <td>-5</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2021-01-10</th>\n",
       "      <td>-0.210608</td>\n",
       "      <td>-1.971006</td>\n",
       "      <td>-0.692841</td>\n",
       "      <td>-5</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                   a         b         c  d    f\n",
       "2021-01-01  0.000000  0.000000 -1.495480 -5 -1.0\n",
       "2021-01-02 -0.731000 -1.349302 -1.059063 -5 -2.0\n",
       "2021-01-03 -1.902861 -0.337020 -0.344063 -5 -3.0\n",
       "2021-01-04 -0.769069 -1.190182 -0.289454 -5 -4.0\n",
       "2021-01-05 -1.318712 -0.509324 -0.663939 -5 -5.0\n",
       "2021-01-06 -0.501530 -0.585483 -1.450538 -5 -6.0\n",
       "2021-01-07 -2.224210 -1.439403 -1.494122 -5  NaN\n",
       "2021-01-08 -0.761851 -0.448527 -0.440605 -5  NaN\n",
       "2021-01-09 -0.783171 -0.270960 -1.189345 -5  NaN\n",
       "2021-01-10 -0.210608 -1.971006 -0.692841 -5  NaN"
      ]
     },
     "execution_count": 78,
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
   "cell_type": "code",
   "execution_count": 79,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['1', '2', '3', '4']"
      ]
     },
     "execution_count": 79,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "l = list(\"1234\")\n",
    "l"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 80,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[5, 5, '3', '4']"
      ]
     },
     "execution_count": 80,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "l[0]=l[1]=5\n",
    "l"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 81,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[5, 5, '3', '4']\n"
     ]
    }
   ],
   "source": [
    "print(l)"
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
