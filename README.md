## 快速排序 时间O(nlogn)，空间O(logn)，不稳定


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
