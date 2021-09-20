{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# List Comprehensions\n",
    "\n",
    "List comprehensions offer a convenient way to create lists out of iterables, filtering or applying functions on elements."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- Creating a list using a list comprehension:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[1, 2, 3, 4]\n"
     ]
    }
   ],
   "source": [
    "sample_list = [1, 2, 3, 4]\n",
    "\n",
    "list_comp = [i for i in sample_list]\n",
    "print(list_comp)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- List comprehensions can be used on any iterable or generator:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['s', 't', 'r', 'i', 'n', 'g']\n"
     ]
    }
   ],
   "source": [
    "s = 'string'\n",
    "\n",
    "list_comp = [letter for letter in s]\n",
    "print(list_comp)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[1, 3, 9, 17]\n"
     ]
    }
   ],
   "source": [
    "sample_set = {1, 3, 9 ,17}\n",
    "\n",
    "list_comp = [element for element in sample_set]\n",
    "print(list_comp)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- Create a list of integers from 0 to 9, using a list comprehension and `range()`"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "integers = [i for i in range(0,10)]\n",
    "integers"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Statements can be applied to each element using a list comprehension."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[3, 4, 5]"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "[i+2 for i in [1, 2, 3]]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Applying functions using list comprehensions:"
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
       "['xof', 'kciuq', 'nworb']"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "reverser = lambda x: x[::-1]\n",
    "\n",
    "[reverser(i) for i in ['fox', 'quick', 'brown']]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "List comprehensions can be used with conditions to filter out certain elements.  \n",
    "- Create a list of elements less than 24:"
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
       "[12, 11, 6, 9]"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "grades = [12, 24, 53, 11, 6, 9, 25]\n",
    "\n",
    "[grade for grade in grades if grade < 24]"
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
       "[55, 65, 80, 75, 60]"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "grades = [55, 65, 40, 80, 75, 60, 30, 25]\n",
    "[successfull_grade for successfull_grade in grades if successfull_grade>50]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- Given the marks students got from a test, how many students got higher than 70?"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[82, 85, 91]\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "3"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "marks = [31, 6, 14, 43, 39, 49, 66, 24, 40, 82, 56, 30, 29, 1, 85, 45, 12, 91, 53, 29]\n",
    "\n",
    "# Write your code here\n",
    "suc_grade = [grade for grade in marks if grade>70]\n",
    "print(suc_grade)\n",
    "len(suc_grade)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Nested list comprehensions can be used for combining elements:\n",
    "- Create a list of all possible outcomes of two fair dice. "
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
       "[(1, 1),\n",
       " (1, 2),\n",
       " (1, 3),\n",
       " (1, 4),\n",
       " (1, 5),\n",
       " (1, 6),\n",
       " (2, 1),\n",
       " (2, 2),\n",
       " (2, 3),\n",
       " (2, 4)]"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "die1 = range(1,7)\n",
    "die2 = range(1,7)\n",
    "\n",
    "outcomes = [(i,j) for i in die1 for j in die2]\n",
    "# Print first few outcomes\n",
    "outcomes[:10]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Given below are the teams in Champions League Group D:\n",
    "\n",
    "|Team|Country|\n",
    "|---|---|\n",
    "|Liverpool|England|\n",
    "|Galatasaray|Turkey|\n",
    "|CSKA Moscow|Russia|\n",
    "|Levski Sofia|Bulgaria|\n",
    "\n",
    "- Create a list of all home and away matches in this group:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[('Liverpool', 'Galatasaray'),\n",
       " ('Liverpool', 'CSKA Moscow'),\n",
       " ('Liverpool', 'Levski Sofia'),\n",
       " ('Galatasaray', 'Liverpool'),\n",
       " ('Galatasaray', 'CSKA Moscow'),\n",
       " ('Galatasaray', 'Levski Sofia'),\n",
       " ('CSKA Moscow', 'Liverpool'),\n",
       " ('CSKA Moscow', 'Galatasaray'),\n",
       " ('CSKA Moscow', 'Levski Sofia'),\n",
       " ('Levski Sofia', 'Liverpool'),\n",
       " ('Levski Sofia', 'Galatasaray'),\n",
       " ('Levski Sofia', 'CSKA Moscow')]"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "group_d = ['Liverpool', 'Galatasaray', 'CSKA Moscow', 'Levski Sofia']\n",
    "[(e, k) for e in group_d for k in group_d if e != k]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- Given two lists of equal length, perform element-wise addition.\n",
    "\n",
    "Sample Input:\n",
    "```\n",
    "l1 = [1, 2, 3]\n",
    "l2 = [2, 4, 8]\n",
    "```\n",
    "Sample Output:\n",
    "```\n",
    "[3, 6, 11]\n",
    "```"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[3, 6, 11]"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "l1 = [1, 2, 3]\n",
    "l2 = [2, 4, 8]\n",
    "result = [l1[i] + l2[i] for i in range(len(l1))]\n",
    "result"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[45, -20, 5, 12, 105]"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "l1 = [16, 24, 5, 11, 92]\n",
    "l2 = [29, -44, 0, 1, 13]\n",
    "\n",
    "element_wise = [l1[i] + l2[i] for i in range(len(l1))]\n",
    "element_wise"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- Define a function using list comprehensions to calculate Mean Squared Error between two lists.  \n",
    "Sample Input:\n",
    "```\n",
    "actual = [9, 6, 4]\n",
    "predicted = [11, 4, 5]\n",
    "mse(actual, predicted)\n",
    "```\n",
    "$$MSE = \\frac{1}{m}\\sum_m{(x_i-y_i)^2}$$\n",
    "$\\frac{(9-11)^2 + (6-4)^2 + (4-5)^2}{3}$  \n",
    "Sample Output:\n",
    "```\n",
    "3.0\n",
    "```"
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
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "def mse(actual, predicted):\n",
    "    # Write your code here. Replace None with your calculated MSE\n",
    "    return None"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [],
   "source": [
    "def mse(actual, predicted):\n",
    "    mean_square = [((actual[i] - predicted[i])**2)  / len(actual) for i in range(len(actual))]\n",
    "    return sum(mean_square)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [],
   "source": [
    "actual = [9, 6, 4]\n",
    "predicted = [11, 4, 5]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "3.0"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "mse(actual, predicted)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1.7933312499999976"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "actual = [19.49, 18.32, 13.16, 8.41, 6.71, 25.87, 19.26, 39.4, \n",
    "          3.13, 11.78, 0.36, 36.13, 1.07, 7.27, 39.98, 27.43]\n",
    "predicted = [20.63, 18.67, 14.82, 6.76, 7.13, 27.48, 22.49, 39.68, \n",
    "             1.87, 11.24, -0.27, 36.86, 0.63, 5.28, 39.13, 28.34]\n",
    "mse(actual, predicted)"
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
