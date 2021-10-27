### 排序
## 快速排序 时间O(nlogn)，空间O(logn)，不稳定


```
def quick_sort(a,start,end):
    if start>=end:
        return
    left = start
    right = end
    key = a[start]
    while left != right:
        while a[right]>key and right>left:
            right = right-1
        if right>left:
            a[left],a[right] = a[right],a[left]
            left = left+1
        while a[left]<key and right>left:
            left = left+1
        if right>left:
            a[right],a[left] = a[left],a[right]
            right = right-1
    a[left] = key
    quick_sort(a,start,left-1)
    quick_sort(a,left+1,end)
b=[23,2,4,5,6,45,56,78]
quick_sort(b,0,len(b)-1)
print(b)
```
## 归并排序 时间O(nlogn)，空间O(n)，稳定
```
def merge_sort(a):
    n = len(a)
    if n <=1:
        return a
    mid = n // 2
    left = merge_sort(a[0:mid])
    right = merge_sort(a[mid:])
    i = 0
    j = 0
    result = []
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            result.append(left[i])
            i +=1
        else:
            result.append(right[j])
            j+=+1

    result += left[i:]
    result += right[j:]

    return result


a = [2,5,33,99,23,45,67]
b =merge_sort(a)
print(b)
```
## 堆排序 ##
空间复杂度1，时间复杂度O(nlogn)
```
def heapify(arr, n, i): 
    largest = i  
    l = 2 * i + 1     # left = 2*i + 1 
    r = 2 * i + 2     # right = 2*i + 2 
  
    if l < n and arr[i] < arr[l]: 
        largest = l 
  
    if r < n and arr[largest] < arr[r]: 
        largest = r 
  
    if largest != i: 
        arr[i],arr[largest] = arr[largest],arr[i]  # 交换
  
        heapify(arr, n, largest) 
  
def heapSort(arr): 
    n = len(arr) 
  
    # Build a maxheap. 
    for i in range(n, -1, -1): 
        heapify(arr, n, i) 
  
    # 一个个交换元素
    for i in range(n-1, 0, -1): 
        arr[i], arr[0] = arr[0], arr[i]   # 交换
        heapify(arr, i, 0) 
  
arr = [ 12, 11, 13, 5, 6, 7] 
heapSort(arr) 
n = len(arr) 
print ("排序后") 
for i in range(n): 
    print ("%d" %arr[i])
```
## 冒泡排序 ##
（英语：Bubble Sort）是一种简单的排序算法。它重复地遍历要排序的数列，一次比较两个元素，如果他们的顺序错误就把他们交换过来。遍历数列的工作是重复地进行直到没有再需要交换，也就是说该数列已经排序完成。这个算法的名字由来是因为越小的元素会经由交换慢慢“浮”到数列的顶端。

冒泡排序算法的运作如下：

比较相邻的元素。如果第一个比第二个大（升序），就交换他们两个。
对每一对相邻元素作同样的工作，从开始第一对到结尾的最后一对。这步做完后，最后的元素会是最大的数。
针对所有的元素重复以上的步骤，除了最后一个。
持续每次对越来越少的元素重复上面的步骤，直到没有任何一对数字需要比较。
```
def bubble_sort(alist):
    n = len(alist)
    # i 代表的是第几趟，从1开始，表示第一趟
    for i in range(1, n):
        is_ordered = True
        # j 表示走一趟
        for j in range(n-i):
            if alist[j] > alist[j+1]:
                alist[j], alist[j+1] = alist[j+1], alist[j]
                is_ordered = False
        if is_ordered:
            return

if __name__ == '__main__':
    lis = [9, 11, 2, 2, 1, 20, 13]
    bubble_sort(lis)
    print(lis)
```
## 选择排序 ##
(Selection sort）是一种简单直观的排序算法。它的工作原理如下。首先在未排序序列中找到最小（大）元素，存放到排序序列的起始位置，然后，再从剩余未排序元素中继续寻找最小（大）元素，然后放到已排序序列的末尾。以此类推，直到所有元素均排序完毕。

选择排序的主要优点与数据移动有关。如果某个元素位于正确的最终位置上，则它不会被移动。选择排序每次交换一对元素，它们当中至少有一个将被移到其最终位置上，因此对n个元素的表进行排序总共进行至多n-1次交换。在所有的完全依靠交换去移动元素的排序方法中，选择排序属于非常好的一种。
```
def select_sort(alist):
    """
    选择排序
    :param alist:
    :return:
    """
    n = len(alist)
    for j in range(n-1):
        min_index = j
        for i in range(j + 1, n):
            if alist[min_index] > alist[i]:
                min_index = i
        if min_index != j:
            alist[j], alist[min_index] = alist[min_index], alist[j] 
            
```
## 插入排序 ##
（英语：Insertion Sort）是一种简单直观的排序算法。它的工作原理是通过构建有序序列，对于未排序数据，在已排序序列中从后向前扫描，找到相应位置并插入。插入排序在实现上，在从后向前扫描过程中，需要反复把已排序元素逐步向后挪位，为最新元素提供插入空间。
```
def insert_sort(alist):
    """
    选择排序
    :param alist:
    :return:
    """
    n = len(alist)
    for j in range(1, n):
        i = j
        while i > 0:
            if alist[i] < alist[i-1]:
                alist[i], alist[i-1] = alist[i-1], alist[i]
                i -= 1
            else:
                break
```
## 希尔排序 ##
```
"""
希尔排序是通过步长把原来的序列分为好几部分，每一个部分采用插入排序，然后调整步长，重复这个过程
最坏时间复杂度考虑gap取1，这就是完全的插入排除
"""
def shell_sort(alist):
    n = len(alist)
    gap = n // 2
    # gap 必须能取到1
    while gap > 0:
        # 插入算法，与普通的插入算法的区别就是gap步长
        for j in range(gap, n):
            i = j
            while i > 0:
                if alist[i] < alist[i-gap]:
                    alist[i], alist[i-gap] = alist[i-gap], alist[i]
                    i -= gap
                else:
                    break
        gap //= 2
```
#### 字符串算法 ####
##旋转字符串 ##

给定字符串，要求把字符串前面若干个字符移动到字符串尾部。要求时间复杂度O(n)，空间复杂度O(1)。
如：'abcdefg'前面的2个字符'a'和'b'移到字符串尾部，就是'cdefgab'。
```
def left_shift_one(s):
    slist = list(s)
    temp = slist[0]
    for i in range(1, len(s)):
        slist[i - 1] = slist[i]
    slist[len(s) - 1] = temp
    s = ''.join(slist)
    return s


def left_rotate_str(s, n):
    while n > 0:
        s = left_shift_one(s)
        n -= 1
    return s
```
## 字符串包含 
给定一长一短的两个字符串A，B，假设A长B短，要求判断B是否包含在字符串A中
```
def string_contain(string_a, string_b):
    list_a = sorted(string_a)
    list_b = sorted(string_b)
    pa, pb = 0, 0
    while pb < len(list_b):
        while pa < len(list_a) and (list_a[pa] < list_b[pb]):
            pa += 1
        if (pa >= len(list_a)) or (list_a[pa] > list_b[pb]):
            return False
        pb += 1
    return True
```
## 回文字符串 ##
判断一个字符串正着读和倒着读是否一样，比如：'abcba'即为回文字符串。
```
def is_palindrome(s):
    s = list(s)
    if len(s)>0:
        start, end = 0, len(s)-1
        while start<end:
            if s[start] != s[end]:
                return False
            start += 1
            end -= 1
        return True
    return True
```
### 动态规划 ###
## 普通背包
```
w = [2,3,4,5]
v = [3,4,5,6]
C=8

dp = [[0 for _ in range(C+1)] for _ in range(len(w)+1)]

for i in range(1,len(w)+1):
    for j in range(1,C+1):
        if j < w[i-1]:
            dp[i][j] = dp[i-1][j]
        else:
            dp[i][j] = max(dp[i-1][j],dp[i-1][j-w[i-1]]+v[i-1])

print(dp)
```
## 完全背包
```
w = [2,3,4,5]
v = [3,4,5,6]
C=8
dp = [[0 for _ in range(C+1)] for _ in range(len(w)+1)]

for i in range(1,len(w)+1):
    for j in range(1,C+1):
        if j < w[i-1]:
            dp[i][j] = dp[i-1][j]
        else:
            dp[i][j] = max(dp[i-1][j],dp[i][j-w[i-1]]+v[i-1])

print(dp)
```
## 多重背包
```
w = [2,3,4,5]
v = [3,4,5,6]
m = [2,3,2,3]
C=8
for i in range(0,len(m)):
    while(m[i]):
        w.append(w[i])
        v.append(v[i])
        m[i] = m[i]-1

dp = [[0 for _ in range(C+1)] for _ in range(len(w)+1)]
for i in range(1,len(w)+1):
    for j in range(1,C+1):
        if j < w[i-1]:
            dp[i][j] = dp[i-1][j]
        else:
            dp[i][j] = max(dp[i-1][j],dp[i][j-w[i-1]]+v[i-1])

print(dp)
```
## 最长公共字串 ##
```
def LCS(string1,string2):
    len1 = len(string1)
    len2 = len(string2)
    res = [[0 for i in range(len1+1)] for j in range(len2+1)]
    for i in range(1,len2+1):
        for j in range(1,len1+1):
            if string2[i-1] == string1[j-1]:
                res[i][j] = res[i-1][j-1]+1
            else:
                res[i][j] = max(res[i-1][j],res[i][j-1])
    return res,res[-1][-1]
print(LCS("helloworld","loop"))
```
## 最长公共子序列 ##
```
def lengthOfLIS(self, nums: List[int]) -> int: # 没有空数组
        max_len = 1
        nums_len = len(nums)
        dp = [1] * nums_len  # dp[i] 表示包含序号i元素的最长上升子序列的长度
        for i in range(1,nums_len):
            for j in range(i):
                if nums[j]<nums[i]:
                    dp[i] = max(dp[i],dp[j]+1)
            max_len = max(max_len,dp[i])
        return max_len
```
## 最长递增连续子序列
```
def findlcis(alist):
	n=o
	res=0
	for i in range(len(alist)):
		if alist[i]>alist[i-1]:
			n+=1
			res=max(res,n)
		else:
			n=1
	return res

```
## 连续子数组最大和 ##
```
def FindGreatestSumOfSubArray(self, array):
        # write code here
        dp = [i for i in array]
        for i in range(1,len(array)):
            dp[i] = max(dp[i-1]+array[i],array[i])
        return max(dp)
```
### 二叉树 ###
## 二叉树的深度 ##
```
class TreeNode:
    def __init__(self, x,leftnode = None,rightnode =None):
        self.val = x
        self.left = leftnode
        self.right = rightnode
        
Tree1 = TreeNode(8,TreeNode(6,TreeNode(5),TreeNode(7)),TreeNode(10,TreeNode(9),TreeNode(11)))

def TreeDepth(self, pRoot):
        # write code here
        if not pRoot:
            return 0
        a = 1+self.TreeDepth(pRoot.left)
        b = 1+self.TreeDepth(pRoot.right)
        return max(a,b)
```
## 二叉搜索树的第k个结点 ##
```
class TreeNode:
    def __init__(self, x,leftnode = None,rightnode =None):
        self.val = x
        self.left = leftnode
        self.right = rightnode
        
Tree1 = TreeNode(8,TreeNode(6,TreeNode(5),TreeNode(7)),TreeNode(10,TreeNode(9),TreeNode(11)))

def __init__(self):
        self.index = 0
def KthNode(self,pRoot,k):
        if not pRoot or not k: return None
        res = []
        def mid_search(cur):
            if not cur: return
            mid_search(cur.left)
            self.index+=1
            res.append(cur)
            mid_search(cur.right)
        mid_search(pRoot)
        if self.index<k: return None
        return res[k-1]
```
## 二叉树的镜像 ##
```
class TreeNode:
    def __init__(self, x,leftnode = None,rightnode =None):
        self.val = x
        self.left = leftnode
        self.right = rightnode
        
Tree1 = TreeNode(8,TreeNode(6,TreeNode(5),TreeNode(7)),TreeNode(10,TreeNode(9),TreeNode(11)))
def Mirror(self, pRoot ):
        # write code here
        if pRoot != None:
            a = pRoot.left
            pRoot.left = pRoot.right
            pRoot.right = a
            self.Mirror(pRoot.left)
            self.Mirror(pRoot.right)
        return pRoot
```
## 从上往下打印二叉树 ##
```
class TreeNode:
    def __init__(self, x,leftnode = None,rightnode =None):
        self.val = x
        self.left = leftnode
        self.right = rightnode
        
Tree1 = TreeNode(8,TreeNode(6,TreeNode(5),TreeNode(7)),TreeNode(10,TreeNode(9),TreeNode(11)))
def PrintFromTopToBottom(self, root):
        # write code here
        res,queue=[],[]
        if root == None:
            return res
        queue.append(root)
        while queue:
            node = queue.pop(0)
            res.append(node.val)
            if node.left!=None:
                queue.append(node.left)
            if node.right!=None:
                queue.append(node.right)
        return res
```
## 二叉搜索树与双向链表 ##
```
class TreeNode:
    def __init__(self, x,leftnode = None,rightnode =None):
        self.val = x
        self.left = leftnode
        self.right = rightnode
        
Tree1 = TreeNode(8,TreeNode(6,TreeNode(5),TreeNode(7)),TreeNode(10,TreeNode(9),TreeNode(11)))
def midTraversal(self,root):
        if not root:
            return []
        self.midTraversal(root.left)
        self.arr.append(root)
        self.midTraversal(root.right)
        
def Convert(self , pRootOfTree ):
        pLast = None
        if pRootOfTree == None:
            return pRootOfTree
        self.arr = []
        self.midTraversal(pRootOfTree)
        for i,v in enumerate(self.arr[:-1]):
            v.right = self.arr[i+1]
            self.arr[i+1].left = v
        return self.arr[0]
```
##  二叉搜索树的后序遍历序列 ##
```
def VerifySquenceOfBST(self, sequence):
        # write code here
        l=len(sequence)
        if l == 0 :
            return False
        index = 0
        for i in range(l):
            if sequence[i]>sequence[-1]:
                index=i
                break
                 
        for j in range(i,l):
            if sequence[j]<sequence[-1]:
                return False
            
        left = True
        right = True
        if len(sequence[:index]) >0:
            left = self.VerifySquenceOfBST(sequence[:index])
        if len(sequence[index:-1]) >0:
            right = self.VerifySquenceOfBST(sequence[index:-1])
            
        return left and right
```
### 栈和队列 
```
class CQueue(object):

    def __init__(self):
        self.s1, self.s2 = [],[]

    def appendTail(self, value):
        """
        :type value: int
        :rtype: None
        """
        self.s1.append(value)

    def deleteHead(self):
        """
        :rtype: int
        """
        if self.s2:
            return self.s2.pop()
        if not self.s1:
            return -1

        while self.s1:
            self.s2.append(self.s1.pop())
        return self.s2.pop()
```
## 包含min函数的栈 ##
```
# -*- coding:utf-8 -*-
class Solution:
    stack1=[]
    stack2=[]    
    def push(self, node):
        # write code here
        self.stack1.append(node)
        if len(self.stack2)==0:
            self.stack2.append(node)
        else:
            if node>self.min():
                self.stack2.append(self.min())
            else:
                self.stack2.append(node)
    def pop(self):
        # write code here
        self.stack1.pop()
        self.stack2.pop()        
    def top(self):
        # write code here
        return self.stack1[len(self.stack1)-1]
    def min(self):
        # write code here
        return self.stack2[len(self.stack2)-1]
```
## 滑动窗口最大值 ##
```
def maxInWindows(self, num, size):
        # write code here
        arr = []
        ans = []
        if num == [] or size > len(num) or size == 0:
            return []
        for i in range(0,len(num) - size + 1):
            arr = num[i:i+size]
            print(arr)
            ans.append(max(arr))
            arr.clear()
        return ans
```
## 替换空格 ##
```
class Solution:
    def replaceSpace(self, s: str) -> str:
        res = []
        for c in s:
            if c == ' ': 
                res.append("%20")
            else: 
                res.append(c)
        return "" .join(res)
```
## 二维矩阵的查找 ##
```
def Find(self, target, array):
        # write code here
        if array==None or array[0]==None:
            return False
        rows,cols=len(array),len(array[0])
        i=0
        j=cols-1
 
        while(i>=0 and i<rows and j>=0 and j<cols):
            if array[i][j]==target:
                return True
            if array[i][j]>target:
                j=j-1
            elif array[i][j]<target:
                i+=1
 
 
        return False
```
## 数组中数字出现的次数 ##
```
class Solution(object):
    def singleNumbers(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        tmp=0
        #首先自遍历异或操作得出一个值
        for i in nums:
            tmp^=i
            print tmp
        #求这个值右边第一个1的值
        num=1
        while not(tmp&num):
            num=num<<1
        result1=0
        result2=0
        #根据这个值对两个数组进行划分
        for i in nums:
            if (i&num):
                result1^=i#自身求异或去掉重复元素
            else:
                result2^=i
        return [result1,result2]
```
## 非极大值抑制 ##

```
import numpy as np
def py_nms(dets, thresh):
    """Pure Python NMS baseline."""
    # x1、y1、x2、y2、以及score赋值
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    # 每一个候选框的面积
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # order是按照score降序排序的
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        # 计算当前概率最大矩形框与其他矩形框的相交框的坐标，会用到numpy的broadcast机制，得到的是向量
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        # 计算相交框的面积,注意矩形框不相交时w或h算出来会是负数，用0代替
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        # 计算重叠度IOU：重叠面积/（面积1+面积2-重叠面积）
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        # 找到重叠度不高于阈值的矩形框索引
        inds = np.where(ovr <= thresh)[0]
        # 将order序列更新，由于前面得到的矩形框索引要比矩形框在原order序列中的索引小1，所以要把这个1加回来
        order = order[inds + 1]
    return keep
# test
if __name__ == "__main__":
    dets = np.array([[30, 20, 230, 200, 1],
                     [50, 50, 260, 220, 0.9],
                     [210, 30, 420, 5, 0.8],
                     [430, 280, 460, 360, 0.7]])
    thresh = 0.35
    keep_dets = py_nms(dets, thresh)
    print(keep_dets)
    print(dets[keep_dets])
    
   
```
## kMeans ##

所以 K-means 的算法步骤为：

选择初始化的 k 个样本作为初始聚类中心 [公式] ；
针对数据集中每个样本 [公式] 计算它到 k 个聚类中心的距离并将其分到距离最小的聚类中心所对应的类中；
针对每个类别 [公式] ，重新计算它的聚类中心 [公式] （即属于该类的所有样本的质心）；
重复上面 2 3 两步操作，直到达到某个中止条件（迭代次数、最小误差变化等）。
```
from numpy import *
from sklearn.datasets import load_iris

def distEclud(vecA,vecB):
    return sqrt(sum(power(vecA-vecB,2)))

def randCent(dataSet,k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k,n)))
    for j in range(n):
        minJ = min(dataSet[:,j])
        maxJ = max(dataSet[:,j])
        rangeJ = float(maxJ-minJ)
        centroids[:,j] = minJ + rangeJ* random.rand(k,1)
    return centroids

def kMeans(dataSet,k,distMeas = distEclud,createCent = randCent):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2)))
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = inf
            minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j,:],dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i,0] != minIndex:
                clusterChanged = True

            clusterAssment[i,:] = minIndex,minDist**2
        print(centroids)

        for cent in range(k):
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]

            centroids[cent,:] = mean(ptsInClust,axis=0)

    return centroids,clusterAssment


iris = load_iris()
X = iris.data[:]
myCentroids,clustAssing = kMeans(X,4)
print (myCentroids)
print (clustAssing)

```
## 计算Iou ##
```
def calIOU_V1(rec1, rec2):
    """
    computing IoU
    :param rec1: (y0, x0, y1, x1), which reflects
            (top, left, bottom, right)
    :param rec2: (y0, x0, y1, x1)
    :return: scala value of IoU
    """
    # cx1 = rec1[0]
    # cy1 = rec1[1]
    # cx2 = rec1[2]
    # cy2 = rec1[3]
    # gx1 = rec2[0]
    # gy1 = rec2[1]
    # gx2 = rec2[2]
    # gy2 = rec2[3]
    cx1, cy1, cx2, cy2 = rec1
    gx1, gy1, gx2, gy2 = rec2
    # 计算每个矩形的面积
    S_rec1 = (cx2 - cx1) * (cy2 - cy1)  # C的面积
    S_rec2 = (gx2 - gx1) * (gy2 - gy1)  # G的面积

    # 计算相交矩形
    x1 = max(cx1, gx1)
    y1 = max(cy1, gy1)
    x2 = min(cx2, gx2)
    y2 = min(cy2, gy2)

    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    area = w * h  # C∩G的面积

    iou = area / (S_rec1 + S_rec2 - area)
    return iou


if __name__ == '__main__':
    rect1 = (661, 27, 679, 47)
    # (top, left, bottom, right)
    rect2 = (662, 27, 682, 47)
    iou = calIOU_V1(rect1, rect2)
    print(iou)
    
```
