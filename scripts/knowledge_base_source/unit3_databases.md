---
title: "CS201 Unit 3: Database Systems"
subject: Database Systems
unit: 3
---

# Unit 3: Database Systems

**CS201 Systems Fundamentals** · Northbridge Institute, School of Computing
Course notes by Dr. Asha Raman · Edition 2026

## Learning outcomes

By the end of this unit you should be able to:

1. Explain the advantages of a DBMS over plain files.
2. Use the vocabulary of the relational model, including the different kinds of keys.
3. Write SQL queries with joins, grouping and filtering.
4. Normalise a table to third normal form, explaining each step.
5. Define the ACID properties and explain concurrency anomalies and isolation levels.
6. Explain how indexes speed up queries and what they cost.

## 3.1 Why use a database management system?

A **database** is an organised collection of related data. A **Database Management System (DBMS)** is the software that stores, retrieves and protects it. Before DBMSs, applications kept data in their own files, which caused real problems:

- **Redundancy and inconsistency:** the same data copied in many files, with copies drifting apart.
- **Difficult access:** every new question needed a new program.
- **Integrity problems:** rules such as "a balance cannot be negative" were buried in application code.
- **Concurrency problems:** two users updating the same file could overwrite each other's changes.
- **Security and recovery:** no fine-grained access control and no clean way to recover after a crash.

A DBMS solves these with a declarative query language (SQL), integrity constraints, transactions, access control and recovery.

## 3.2 The relational model

In the **relational model**, data is stored in **relations**, which we usually call tables.

| Formal term | Everyday term |
|---|---|
| Relation | Table |
| Tuple | Row (record) |
| Attribute | Column (field) |
| Domain | The set of allowed values for a column |

### Keys

- A **super key** is any set of columns that uniquely identifies each row.
- A **candidate key** is a *minimal* super key: no column can be removed and still guarantee uniqueness.
- The **primary key** is the candidate key chosen to identify rows. It cannot be NULL.
- A **foreign key** is a column (or set of columns) in one table that refers to the primary key of another table, linking the two.

### Integrity constraints

- **Entity integrity:** no part of a primary key may be NULL.
- **Referential integrity:** every foreign-key value must match an existing primary-key value in the referenced table (or be NULL, if allowed). You cannot enrol a student who does not exist.

## 3.3 SQL

**SQL (Structured Query Language)** has two main parts. The **Data Definition Language (DDL)** defines structure (`CREATE TABLE`, `ALTER TABLE`, `DROP TABLE`). The **Data Manipulation Language (DML)** works with the data (`SELECT`, `INSERT`, `UPDATE`, `DELETE`).

The examples below use two tables:

- `Students(student_id, name, programme)`
- `Enrolments(student_id, course_code, grade)`, where `student_id` is a foreign key to `Students`

**Filtering:**

```sql
SELECT name FROM Students WHERE programme = 'Computer Science';
```

**Inner join** (only rows that match in both tables):

```sql
SELECT s.name, e.course_code, e.grade
FROM Students s
JOIN Enrolments e ON s.student_id = e.student_id;
```

**Left join** (every student, with NULLs for students who are not enrolled in anything):

```sql
SELECT s.name, e.course_code
FROM Students s
LEFT JOIN Enrolments e ON s.student_id = e.student_id;
```

**Grouping and aggregates:** count the students in each course, keeping only courses with more than 30:

```sql
SELECT course_code, COUNT(*) AS num_students
FROM Enrolments
GROUP BY course_code
HAVING COUNT(*) > 30;
```

`WHERE` filters **rows before** grouping; `HAVING` filters **groups after** grouping.

## 3.4 Entity–relationship modelling

Databases are usually designed with an **entity–relationship (ER) model** before any tables are created. **Entities** are things we store data about (Student, Course); **attributes** describe them; **relationships** connect them (a Student *enrols in* a Course).

Relationships have a **cardinality**:

- **One-to-one (1:1):** each student has one student card.
- **One-to-many (1:N):** one lecturer teaches many courses.
- **Many-to-many (M:N):** students take many courses, and each course has many students.

A many-to-many relationship cannot be stored directly in two tables; it becomes a separate **junction (associative) table**. `Enrolments` above is exactly that: its primary key is the combination `(student_id, course_code)`.

## 3.5 Normalisation

A badly designed table repeats data, which causes three kinds of **anomaly**:

- **Update anomaly:** a fact stored in many rows must be changed in all of them, or the data becomes inconsistent.
- **Insertion anomaly:** we cannot record one fact without inventing another (for example, we cannot add a new course until a student enrols).
- **Deletion anomaly:** deleting one fact accidentally deletes another (removing the last student in a course loses the course's details).

**Normalisation** removes these anomalies by splitting tables according to **functional dependencies**. A functional dependency **X → Y** means that the value of X determines the value of Y. For example, `student_id → name`.

### The normal forms

- **First normal form (1NF):** every column holds **atomic** (indivisible) values: no lists or repeating groups in a cell.
- **Second normal form (2NF):** in 1NF, and **no partial dependency**: no non-key column depends on only *part* of a composite primary key.
- **Third normal form (3NF):** in 2NF, and **no transitive dependency**: no non-key column depends on another non-key column.
- **Boyce–Codd normal form (BCNF):** a stricter 3NF in which, for every dependency X → Y, X must be a super key.

A useful summary: in 3NF, every non-key attribute depends on "the key, the whole key, and nothing but the key".

### Worked example

Consider one table: `Enrolment(student_id, course_code, student_name, course_title, lecturer, lecturer_office, grade)`, with primary key `(student_id, course_code)`.

1. **Partial dependencies** (violate 2NF): `student_id → student_name` and `course_code → course_title, lecturer`. Each depends on only part of the key.
2. Move them out: `Students(student_id, student_name)`, `Courses(course_code, course_title, lecturer, lecturer_office)` and `Enrolments(student_id, course_code, grade)`. The schema is now in 2NF.
3. **Transitive dependency** (violates 3NF): in `Courses`, `course_code → lecturer` and `lecturer → lecturer_office`, so the office depends on the key only *through* the lecturer.
4. Split again: `Courses(course_code, course_title, lecturer)` and `Lecturers(lecturer, lecturer_office)`. The schema is now in **3NF**.

### Common misconceptions

- *"Normalisation is always good."* Highly normalised schemas need more joins. Analytical systems sometimes **denormalise** on purpose to speed up reads.
- *"2NF matters for every table."* A table with a single-column primary key cannot have partial dependencies, so if it is in 1NF it is automatically in 2NF.

## 3.6 Transactions and concurrency

A **transaction** is a sequence of operations that forms one logical unit of work, such as transferring money between two accounts. Transactions guarantee the **ACID** properties:

- **Atomicity:** all of the transaction happens, or none of it does. If the transfer fails halfway, the debit is undone.
- **Consistency:** a transaction takes the database from one valid state to another, respecting every constraint.
- **Isolation:** concurrent transactions do not interfere; the result is as if they ran one after another.
- **Durability:** once a transaction **commits**, its changes survive crashes and power failures.

### Concurrency anomalies

When isolation is weakened for performance, these anomalies can appear:

| Anomaly | What happens |
|---|---|
| Dirty read | A transaction reads data written by another transaction that has not committed yet (and may roll back) |
| Non-repeatable read | A transaction reads the same row twice and gets different values, because another transaction updated it in between |
| Phantom read | A transaction re-runs a query and finds new rows inserted by another transaction |
| Lost update | Two transactions read the same value and both write, so one update overwrites the other |

### Isolation levels

| Isolation level | Dirty read | Non-repeatable read | Phantom read |
|---|---|---|---|
| Read uncommitted | Possible | Possible | Possible |
| Read committed | Prevented | Possible | Possible |
| Repeatable read | Prevented | Prevented | Possible |
| Serializable | Prevented | Prevented | Prevented |

Higher isolation is safer but allows less concurrency.

### Locking

Many DBMSs use locks. A **shared (S) lock** allows reading, and many transactions can hold one at the same time. An **exclusive (X) lock** allows writing, and only one transaction can hold it. Under **two-phase locking (2PL)**, a transaction acquires all its locks in a *growing phase* and releases them in a *shrinking phase*, never acquiring a lock after releasing one. 2PL guarantees serializable schedules, but it can cause **deadlocks** between transactions, exactly the problem studied in Unit 1. Databases detect such cycles and abort one transaction as the victim.

## 3.7 Indexing

Without an index, finding a row means scanning the whole table: a **full table scan**, O(n). An **index** is a separate data structure that maps column values to row locations, like the index at the back of a book.

- **B+ tree index:** a balanced tree in which all values are stored in the leaves, linked in order. Lookups take O(log n), and the linked leaves make **range queries** (`WHERE marks BETWEEN 60 AND 80`) efficient. It is the default index in most relational databases.
- **Hash index:** hashes the key to find the row in roughly O(1) for **equality** lookups, but cannot help with ranges or sorting.

Indexes are not free: they take storage space, and **every INSERT, UPDATE and DELETE must also update every index** on the table. Index the columns you search, join and sort on often, not every column.

## 3.8 Recovery

To guarantee atomicity and durability, databases use **write-ahead logging (WAL)**: every change is first written to a log on stable storage *before* the data pages themselves are changed. After a crash, the DBMS **redoes** the changes of committed transactions and **undoes** the changes of uncommitted ones, using the log. **Checkpoints** limit how much of the log must be replayed.

## 3.9 Beyond relational: NoSQL in brief

**NoSQL** databases relax the relational model to scale out across many machines or to fit particular data shapes: **key–value stores** (very fast lookups by key), **document stores** (flexible JSON-like documents), **column-family stores** and **graph databases**. The **CAP theorem** states that when a network **partition** occurs, a distributed system must choose between **consistency** (every read sees the latest write) and **availability** (every request gets a response).

## Key terms

| Term | Definition |
|---|---|
| DBMS | Software that stores, retrieves and protects data |
| Primary key | The chosen minimal set of columns that uniquely identifies each row; never NULL |
| Foreign key | A column referring to the primary key of another table |
| Referential integrity | Every foreign-key value must match an existing referenced row |
| Functional dependency | X → Y: the value of X determines the value of Y |
| 3NF | No partial and no transitive dependencies on the key |
| Transaction | A sequence of operations treated as one unit of work |
| ACID | Atomicity, Consistency, Isolation, Durability |
| Dirty read | Reading uncommitted data written by another transaction |
| Two-phase locking | Acquire all locks before releasing any; guarantees serializability |
| B+ tree | Balanced tree index supporting fast lookups and range queries |
| Write-ahead logging | Logging changes before applying them, enabling recovery |

## Unit summary

- A DBMS avoids the redundancy, integrity, concurrency and recovery problems of plain files.
- Primary keys identify rows; foreign keys link tables and enforce referential integrity.
- SQL joins combine tables; `WHERE` filters rows before grouping, `HAVING` filters groups after it.
- Normalisation to 3NF removes partial and transitive dependencies and the anomalies they cause.
- Transactions provide ACID guarantees; isolation levels trade safety against concurrency.
- B+ tree indexes make lookups and range queries fast at the cost of slower writes and extra storage.
