--Part 2, Question 1)
SELECT OFFERS.OFFER_ID, OFFERS.ACTION, OFFERS.O_DATE 
FROM OFFERS, BRANCHES, OFFERS_BRANCHES
WHERE OFFERS.OFFER_ID = OFFERS_BRANCHES.OFFER_ID AND OFFERS_BRANCHES.BRANCH_ID = BRANCHES.BRANCH_ID AND BRANCHES.BRANCH_NAME = 'LoyaltyFirst';

--Part 2, Question 2)
SELECT TRANSACTIONS.TREF, TRANSACTIONS.AMOUNT, TRANSACTIONS.T_POINTS, TRANSACTIONS.T_DATE
FROM TRANSACTIONS, CUSTOMERS
WHERE TRANSACTIONS.cid = CUSTOMERS.cid AND CUSTOMERS.cname = 'John Smith';

--Part 2, Question 3)
SELECT BRANCHES.BRANCH_ID, COUNT(OFFERS.OFFER_ID)
FROM BRANCHES, OFFERS, OFFERS_BRANCHES
WHERE OFFERS.OFFER_ID = OFFERS_BRANCHES.OFFER_ID AND OFFERS_BRANCHES.BRANCH_ID = BRANCHES.BRANCH_ID
GROUP BY(BRANCHES.BRANCH_ID);

--Part 2, Question 4)
SELECT BRANCHES.BRANCH_NAME, COUNT(OFFERS.OFFER_ID)
FROM BRANCHES, OFFERS, OFFERS_BRANCHES
WHERE OFFERS.OFFER_ID = OFFERS_BRANCHES.OFFER_ID AND OFFERS_BRANCHES.BRANCH_ID = BRANCHES.BRANCH_ID
GROUP BY(BRANCHES.BRANCH_NAME);

--Part 2, Question 5)
SELECT TRANSACTIONS.TREF, TRANSACTIONS.T_DATE, TRANSACTIONS.T_TIME, TRANSACTIONS.AMOUNT, TRANSACTIONS.T_POINTS, PRODUCTS.PROD_ID, PRODUCTS.PROD_NAME, TRANSACTIONS_PRODUCTS.QUANTITY, PRODUCTS.PRICE, PRODUCTS.PROD_POINTS
FROM TRANSACTIONS, PRODUCTS, TRANSACTIONS_PRODUCTS
WHERE TRANSACTIONS.TREF = TRANSACTIONS_PRODUCTS.TREF AND PRODUCTS.PROD_ID = TRANSACTIONS_PRODUCTS.PROD_ID AND TRANSACTIONS.TREF = '9091978443287546211';

--Part 2, Question 6)
SELECT COUNT(*)
FROM CARDS
WHERE CARDS.EXP_DATE < TRUNC(SYSDATE);

--Part 2, Question 7)
SELECT CUSTOMERS.CID 
FROM CUSTOMERS, CARDS
WHERE CUSTOMERS.CID = CARDS.CID AND CARDS.EXP_DATE < TRUNC(SYSDATE)
GROUP BY CUSTOMERS.CID
ORDER BY COUNT(*) DESC
OFFSET 0 ROWS 
FETCH NEXT 1 ROWS ONLY;

--Part 2, Question 8)
SELECT PRIZES.PRIZE_ID, PRIZES.P_DESCRIPTION, CUSTOMERS.CNAME, REDEMPTION_HISTORY.CENTER_ID, REDEMPTION_HISTORY.QUANTITY
FROM PRIZES, CUSTOMERS, REDEMPTION_HISTORY
WHERE PRIZES.PRIZE_ID = REDEMPTION_HISTORY.PRIZE_ID AND CUSTOMERS.CID = REDEMPTION_HISTORY.CID  AND CUSTOMERS.CNAME = 'Mary Smith';

--Part 2, Question 9)
SELECT CNAME, OCCUPATION 
FROM CUSTOMERS, FAMILIES
WHERE CUSTOMERS.FAMILY_ID = FAMILIES.FAMILY_ID AND FAMILIES.FAMILY_NAME = 'Smith';

--Part 2, Question 10)
SELECT SUM(NUM_OF_POINTS)
FROM FAMILIES, POINT_ACCOUNTS
WHERE POINT_ACCOUNTS.FAMILY_ID = FAMILIES.FAMILY_ID AND FAMILIES.FAMILY_NAME = 'Smith';

--Part 2, Question 11)
SELECT CUSTOMERS.CNAME
FROM CUSTOMERS, POINT_ACCOUNTS
WHERE POINT_ACCOUNTS.CID = CUSTOMERS.CID
GROUP BY CUSTOMERS.CNAME, POINT_ACCOUNTS.NUM_OF_POINTS
ORDER BY COUNT(*) DESC
OFFSET 0 ROWS 
FETCH NEXT 1 ROWS ONLY;

--Part 2, Question 12)
SELECT COUNT(QUANTITY)
FROM REDEMPTION_HISTORY
WHERE REDEMPTION_HISTORY.R_DATE = TO_DATE('11-26-2019', 'MM-DD-YYYY');

--Part 2, Question 13)
SELECT COUNT(PRIZE_ID)
FROM REDEMPTION_HISTORY
WHERE REDEMPTION_HISTORY.CID = '29';

--Part 2, Question 14)
SELECT COUNT(CID)
FROM REDEMPTION_HISTORY
WHERE REDEMPTION_HISTORY.CENTER_ID = '9798720907171277284';

--Part 2, Question 15)
SELECT COUNT(PRIZE_ID)
FROM PRIZES;

--Part 2, Question 16)
SELECT CUSTOMERS.CNAME
FROM CUSTOMERS, ADDRESSES
WHERE CUSTOMERS.CID = ADDRESSES.CID AND CUSTOMERS.OCCUPATION = 'Engineer' AND ADDRESSES.CITY = 'Fairfax';

--Part 2, Question 17)
SELECT PRODUCTS.PROD_NAME
FROM PRODUCTS, TRANSACTIONS, TRANSACTIONS_PRODUCTS
WHERE PRODUCTS.PROD_ID NOT IN (SELECT TRANSACTIONS_PRODUCTS.PROD_ID FROM TRANSACTIONS_PRODUCTS);

--Part 2, Question 18)
SELECT PRODUCTS.PROD_NAME
FROM PRODUCTS, TRANSACTIONS_PRODUCTS, TRANSACTIONS, CUSTOMERS
WHERE PRODUCTS.PROD_ID = TRANSACTIONS_PRODUCTS.PROD_ID AND TRANSACTIONS.TREF = TRANSACTIONS_PRODUCTS.TREF AND TRANSACTIONS.CID = CUSTOMERS.CID
GROUP BY (PRODUCTS.PROD_NAME, PRODUCTS.PROD_ID)
ORDER BY COUNT(*) DESC
OFFSET 0 ROWS 
FETCH NEXT 1 ROWS ONLY;
