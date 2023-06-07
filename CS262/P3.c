#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <math.h>


/*Node structure for the linked list.*/
typedef	struct Node{
	int data;
	struct	Node *next;
}ListNode;

/*Initializer function to create a new Linked List.*/
ListNode *newList()
{
    ListNode* node = malloc(sizeof(struct Node));
    node->data = INT_MAX;
    node->next = NULL;
    return node;
}

/*Removes a node from the linked list*/
ListNode *removeNode(ListNode * prev)
{
    ListNode *returnNode = prev->next;
    free(prev->next);
    return returnNode;
}

/*Inserts a node into the linked list*/
ListNode *insertNode(ListNode *prev, int data)
{
    ListNode *returnNode = malloc(sizeof(struct Node));
    returnNode->data = data;
    if(prev->data == INT_MAX)
    {
        returnNode->next = prev;
        prev = returnNode;
    }
    else
    {
        prev->next = returnNode;
    }
    return returnNode;
}

/*Returns the size of the linked list from the given head node.*/
int length(ListNode *head)
{
    int size = 0;
    ListNode *tempNode = malloc(sizeof(struct Node));
    tempNode = head;
    while(tempNode->next != NULL)
    {
        size += 1;
        tempNode = tempNode->next;
    }
    free(tempNode);
    return size;
}

/*Prints the linked list from the given head node.*/
void printList(ListNode *head)
{
    ListNode *tempNode = malloc(sizeof(struct Node));
    tempNode = head;
    while(tempNode != NULL && tempNode->data != INT_MAX)
    {
        printf("%d ", tempNode->data);
        tempNode = tempNode->next;
    }
    printf("\n");
}

/*Prints every value in the given array*/
void printArray(int list[], int size)
{   int i = 0;
    for(i = 0; i < size; i++)
    {
        printf("%d ", list[i]);
    }
    printf("\n");
}

/*Deletes a linked list*/
void deleteList(ListNode *head)
{
    ListNode* prev = head;
    while (head)
    {
        head = head->next;
        free(prev);
        prev = head;
    }
}

/*Finds the maximum value in the array.*/
int findMax(int nums[], int arraySize)
{
    int max = 0;
    int i = 0;
    for(i = 0; i < arraySize; i++)
    {
        if(nums[i] > max)
        {
            max = nums[i];
        }
    }
    
    return max;
}

/*Returns the number of places of the given int.*/
int numPlaces (int n) 
{
    if (n == 0) return 1;
    return floor (log10 (abs(n))) + 1;
}

/*Returns the digit at the given position in the given num*/
int getDigitAtPosition(int num, int position)
{
    if(position != 0)
    {
        return num / pow(10, position);
    }
    else
    {
        return num % 10;
    }
}

/*Converts an array of linked lists to an array*/
void fillArray(int returnArray[], ListNode *arrayOfNodes[])
{
    int arrayIndex = 0;
    int i = 0;
    for(i = 0; i < 10; i++)
    {
        ListNode *tempNode = arrayOfNodes[i];
        while(tempNode != NULL)
        {
            if(tempNode->data != INT_MAX)
            {
                returnArray[arrayIndex] = tempNode->data; 
                arrayIndex = arrayIndex + 1;
            }
            tempNode = tempNode->next;
        }
    }
}

/*Sorts a list of nums into buckets*/
void SortList(int nums[], ListNode *arrayOfNodes[], int arraySize)
{
    int max = findMax(nums, arraySize);
    int recursions = numPlaces(max);
    int index = 0;
    int i = 0;
    int k = 0;
    while(index < recursions)
    {
        for(i = 0; i < arraySize; i++)
        {
            if(arrayOfNodes[getDigitAtPosition(nums[i], index)]->data == INT_MAX)
            {
                arrayOfNodes[getDigitAtPosition(nums[i], index)] = insertNode(arrayOfNodes[getDigitAtPosition(nums[i], index)], nums[i]);
            }
            else
            {
                ListNode *current = arrayOfNodes[getDigitAtPosition(nums[i], index)];
                while(current->next != NULL && current->next->data != INT_MAX)
                {
                    current = current->next;
                }
                
                current->next = insertNode(current, nums[i]);
            }   
        }
        for(k = 0; k < 10; k++)
        {
            printf("%d List contains: ", k);
            printList(arrayOfNodes[k]);
        }
        fillArray(nums, arrayOfNodes);
        printf("Pass %d: ", index);
        printArray(nums, arraySize);
        if(index + 1 != recursions)
        {
	        int j = 0;
            for(j = 0; j < 10; j++)
            {
                deleteList(arrayOfNodes[j]);
                arrayOfNodes[j] = malloc(sizeof(struct Node));
                arrayOfNodes[j] = newList();
            }
        }
        index = index + 1;
    }
    

}

/*Sews all the buckets together into a sorted list*/
ListNode *sewLists(ListNode *nodeHeads[])
{
    int i = 0;
    ListNode *headOfHeads = nodeHeads[0];
    for(i = 0; i < 10; i++)
    {
        ListNode *tempNode = malloc(sizeof(struct Node));
        if(headOfHeads->data == INT_MAX && nodeHeads[i]->data != INT_MAX)
        {
            headOfHeads = nodeHeads[i];
        }
        else
        {
            tempNode = headOfHeads;
            while(tempNode->next != NULL && tempNode->next->data != INT_MAX)
            {
               tempNode = tempNode->next; 
            }
            if(i != 9 && nodeHeads[i+1]->data != INT_MAX)
            {
                tempNode->next = nodeHeads[i+1];
            }
        }
    }
    return headOfHeads;
}


/*Driver code for input validation and program execution*/
int main(int argc, char *argv[])
{
   if(argc != 5)
   {
       printf("Error! Please input four values: A seed for the random number generator, the number of values to be generated, the lower bound of random generator, and the upper bound for the random number generator!");
   }
   else
   {
    int upperBound = atoi(argv[4]);
    int lowerBound = atoi(argv[3]);
    int numValues = atoi(argv[2]);
    int randSeed = atoi(argv[1]);
    int i = 0;
    int k = 0;
    ListNode *nodeHeads[10];
    ListNode* sewnList;
    int *numArray =  malloc(numValues * sizeof(int));;
    if(lowerBound > upperBound)
    {
        printf("Error! Please make sure the lower bound is less than the upper bound!");
        exit(0);
    }
    if (numValues <= 0)
    {
        printf("Error! Cannot run with zero, or less than zero values!");
        exit(0);
    }
    srand(randSeed);
    for(i = 0; i < numValues; i++)
    {
        numArray[i] = (rand() % upperBound) + lowerBound;
    }
    for(k = 0; k < 10; k++)
    {
        nodeHeads[k] = malloc(sizeof(struct Node));
        nodeHeads[k] = newList();
    }
    printf("Unsorted Array: ");
    printArray(numArray, numValues);
    SortList(numArray, nodeHeads, numValues);
    printf("Sorted Array (Taken from lists): ");
    sewnList = sewLists(nodeHeads);
    printList(sewnList);
    free(sewnList);
    free(numArray);
   }
   return 0;
}
