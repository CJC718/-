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
## 堆排序 空间复杂度1，时间复杂度O(nlogn)
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
## 冒泡排序（英语：Bubble Sort）是一种简单的排序算法。它重复地遍历要排序的数列，一次比较两个元素，如果他们的顺序错误就把他们交换过来。遍历数列的工作是重复地进行直到没有再需要交换，也就是说该数列已经排序完成。这个算法的名字由来是因为越小的元素会经由交换慢慢“浮”到数列的顶端。

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
## 选择排序（Selection sort）是一种简单直观的排序算法。它的工作原理如下。首先在未排序序列中找到最小（大）元素，存放到排序序列的起始位置，然后，再从剩余未排序元素中继续寻找最小（大）元素，然后放到已排序序列的末尾。以此类推，直到所有元素均排序完毕。

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
## 插入排序（英语：Insertion Sort）是一种简单直观的排序算法。它的工作原理是通过构建有序序列，对于未排序数据，在已排序序列中从后向前扫描，找到相应位置并插入。插入排序在实现上，在从后向前扫描过程中，需要反复把已排序元素逐步向后挪位，为最新元素提供插入空间。
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
## 希尔排序
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
    
