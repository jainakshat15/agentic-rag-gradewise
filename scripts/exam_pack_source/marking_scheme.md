---
title: "Midterm Examination: Marking Scheme"
header: "CS201 Systems Fundamentals · Midterm Examination · Marking Scheme (CONFIDENTIAL)"
---

# CS201 Midterm: Marking Scheme

**Northbridge Institute, School of Computing** · CS201 Systems Fundamentals · Exam code CS201-MT-2026
Examiner: Dr. Emma Carter · Total: **50 marks** · **CONFIDENTIAL: for markers only**

## How to use this scheme

Each question (or part) is broken into numbered **mark points**. Award each point independently; the marks for a part are the sum of its points and can never exceed the marks printed on the paper. "Accept" lists wording or alternatives that earn the point. "Do not accept" lists common answers that do not. **Keywords** are there to help you find the point quickly: they are *not* a checklist, and an answer that expresses the idea correctly in other words earns the mark (general marking principle 3). Read this scheme together with the *Marking Methodology* document and the general marking principles in the course handbook.

Abbreviations: **ECF** = error carried forward · **OWTTE** = or words to that effect · **½** = half a mark.

---

## Question 1: system calls [3 marks]

| Point | Marks | The candidate... |
|---|---|---|
| 1 | 1 | Defines a system call as a controlled request from a user program to the kernel (operating system) for a privileged service, such as reading a file or creating a process. |
| 2 | 1 | States that applications run in user mode, which is restricted and cannot access hardware or privileged instructions directly, whereas kernel mode is privileged. |
| 3 | 1 | Explains the switch: the CPU enters kernel mode so the kernel can do the work on the program's behalf, then returns to user mode; this protects the system, so a buggy program cannot corrupt the kernel or other programs. |

- **Keywords:** kernel, privileged, user mode, kernel mode, protection.
- **Accept** "trap", "software interrupt" or "controlled entry point into the kernel" for point 1. Accept an example service (open a file, fork a process, send network data) as part of point 1, but an example alone, with no idea of *requesting the kernel*, earns ½.
- **Point 3** needs a *reason* (protection, safety, isolation). "Because it needs to" earns nothing.
- **Do not accept** "a system call is when the computer calls the system" or any answer that describes a function call inside the program.

## Question 2(a): definition of a process [2 marks]

| Point | Marks | The candidate... |
|---|---|---|
| 1 | 1 | States that a process is a program in execution (an active entity), as opposed to a program stored on disk (passive). |
| 2 | 1 | States that a process has its own state or resources: at least TWO of its own address space (code, data, heap, stack), program counter, registers, open files, or a Process Control Block (PCB). |

- **Keywords:** program in execution, address space, program counter, PCB.
- **Partial credit:** only one item of state for point 2 earns ½.
- **Do not accept** "a process is a program" (with no idea of execution) for point 1.

## Question 2(b): process vs thread [3 marks]

Award **1 mark for each correct difference, up to three.** Any three of the following:

| Point | Marks | The candidate... |
|---|---|---|
| 1 | 1 | Gives a first correct difference (see the list below). |
| 2 | 1 | Gives a second, different correct difference. |
| 3 | 1 | Gives a third, different correct difference. |

Acceptable differences:

- **Memory:** a process has its own separate address space; threads of a process share its code, data, heap and open files (each thread keeps its own stack, registers and program counter).
- **Creation cost:** creating a process is expensive (new address space); creating a thread is cheap.
- **Context switch:** switching between threads is faster than switching between processes.
- **Communication:** processes need inter-process communication (pipes, sockets, shared memory); threads communicate directly through shared variables.
- **Isolation:** a crash in one process usually affects only that process; a crash in one thread can bring down the whole process.
- Accept "a thread is part of a process / a process contains one or more threads" as **one** difference.

- **Partial credit:** a difference stated for one side only (for example "threads are faster") earns ½. The two sides must be contrasted for the full mark.
- The same difference written twice in different words earns the mark once.

## Question 3: deadlock [4 marks]

| Point | Marks | The candidate... |
|---|---|---|
| 1 | 2 | Gives a correct definition: a set of processes (or threads) that are all blocked, each waiting for a resource held by another process in the set, so none of them can ever proceed. |
| 2 | 1 | Names a first correct necessary condition: mutual exclusion, hold and wait, no preemption, or circular wait. |
| 3 | 1 | Names a second, different correct necessary condition from the same list. |

- **Keywords:** blocked, waiting, resource, held by another, mutual exclusion, hold and wait, no preemption, circular wait.
- **Partial credit for point 1:** award **1 of the 2 marks** for "processes stuck, each waiting for the other" if the idea of a *resource held by another process* is missing, or if the answer does not make clear that the processes can *never* proceed.
- **Accept** "Coffman conditions" as the collective name, and "circular waiting" or "no pre-emption" spellings.
- The question asks for TWO conditions: if more are given, mark the first two. A wrong condition (for example "starvation") earns nothing for that point.
- **Do not accept** "deadlock is when the computer freezes" or a description of starvation (a process that *could* run but keeps being passed over).

## Question 4(a): race condition [2 marks]

| Point | Marks | The candidate... |
|---|---|---|
| 1 | 1 | States that two or more threads or processes access (and at least one updates) shared data at the same time. |
| 2 | 1 | States that the final result depends on the timing or order in which their steps happen (for example, an increment is lost). |

- **Keywords:** shared data, concurrently, order, timing, lost update.
- **Accept** the `counter = counter + 1` lost-increment example for point 2 if it shows the result depends on the interleaving.

## Question 4(b): semaphores [4 marks]

| Point | Marks | The candidate... |
|---|---|---|
| 1 | 1 | States that a semaphore is an integer variable that, apart from initialisation, is accessed only through two atomic operations. |
| 2 | 1 | Explains wait() (P / down): decrements the value, and the calling thread blocks if the value becomes negative (or if no resource is available). |
| 3 | 1 | Explains signal() (V / up): increments the value and wakes one blocked thread, if any. |
| 4 | 1 | Distinguishes binary (values 0 and 1 only, behaves like a mutex lock) from counting (any non-negative value, controls a resource with several identical instances, for example a pool of 5 connections). |

- **Keywords:** integer, atomic, wait, signal, P, V, block, binary, counting, mutex.
- **Point 1:** the word "atomic" or the idea "indivisible / cannot be interrupted half-way" is needed for the full mark. "A semaphore is a variable used for locking" earns ½.
- **Point 4:** both types must be described. One type only earns ½.
- **Do not accept** "a semaphore is just a lock" for point 4: that is true only for a binary semaphore.

## Question 5: page-table calculation [5 marks]

Correct answers: offset = **13 bits**, page number = **19 bits**, **2^19 = 524,288 pages**, page table = 2^19 × 4 B = 2^21 B = **2 MB**.

| Point | Marks | The candidate... |
|---|---|---|
| 1 | 1 | Finds the offset: 8 KB = 2^13 bytes, so the offset is 13 bits. |
| 2 | 1 | Finds the page-number bits: 32 − 13 = 19 bits. |
| 3 | 1 | Finds the number of pages: 2^19 = 524,288 (accept 512 K or 512 × 1024). |
| 4 | 1 | Method mark: page-table size = number of pages (entries) × size of one entry, i.e. 2^19 × 4 bytes. |
| 5 | 1 | Correct final answer with units: 2^21 bytes = 2 MB (accept 2,097,152 bytes or 2 MiB). |

- **Method marks / ECF:** if the offset is wrong (for example 12 bits from using 4 KB), award points 2, 3, 4 and 5 for answers that are correct *for the candidate's offset* (e.g. 20 bits → 2^20 pages → 4 MB earns points 2, 3, 4 and 5 under ECF).
- **Units:** a final answer of "2" or "2^21" with no unit loses point 5 only (at most one mark per question is lost for units).
- **Answer only:** a correct final answer of 2 MB with no working earns point 5 only.
- Accept 1 KB = 1000 B working only if applied consistently; the expected answer uses 1 KB = 1024 B.

## Question 6(a): TCP three-way handshake [3 marks]

| Point | Marks | The candidate... |
|---|---|---|
| 1 | 1 | Step 1, SYN: the client sends a segment with the SYN flag set (and its initial sequence number). |
| 2 | 1 | Step 2, SYN-ACK: the server replies with SYN and ACK set, acknowledging the client's sequence number and sending its own. |
| 3 | 1 | Step 3, ACK: the client acknowledges the server's sequence number, and the connection is established. |

- **Keywords:** SYN, SYN-ACK, ACK, sequence number, established.
- **Order matters:** if the three steps are named correctly but in the wrong order, award a maximum of 2 marks.
- A bare list "SYN, SYN-ACK, ACK" with no indication of who sends what earns at most **1½** (principle 4: listing without explanation earns at most half).

## Question 6(b): TCP vs UDP [3 marks]

| Point | Marks | The candidate... |
|---|---|---|
| 1 | 1 | Gives a first correct difference between TCP and UDP. |
| 2 | 1 | Gives a second, different correct difference. |
| 3 | 1 | Names a suitable UDP application AND gives a valid reason. |

Acceptable differences: connection-oriented (handshake) vs connectionless · reliable with retransmission vs unreliable · ordered delivery vs no ordering guarantee · flow and congestion control vs none · 20-byte vs 8-byte header · higher overhead vs lower overhead and latency.

- **Point 3:** accept video calls, online games, live streaming, DNS queries or VoIP, with a reason such as "low latency matters more than reliability", "a late packet is useless", or "queries are small and fast". An application with no reason earns ½. Do not accept web browsing, email or file transfer (these use TCP).
- As in Q2(b), a one-sided difference ("UDP is faster") earns ½.

## Question 7(a): usable hosts [2 marks]

| Point | Marks | The candidate... |
|---|---|---|
| 1 | 1 | Method: host bits = 32 − 27 = 5, so the subnet has 2^5 = 32 addresses. |
| 2 | 1 | Subtracts the network and broadcast addresses: 32 − 2 = **30 usable hosts**. |

- **Answer only:** "30" with no working earns point 2 only.
- If the candidate gives 32 (forgetting to subtract 2), award point 1 only.

## Question 7(b): network address, broadcast address and mask [3 marks]

| Point | Marks | The candidate... |
|---|---|---|
| 1 | 1 | Network address **172.16.45.192** (blocks of 32 in the last octet: 0, 32, …, 192, 224, and 200 lies in the 192 block). |
| 2 | 1 | Broadcast address **172.16.45.223** (the last address of the block). |
| 3 | 1 | Subnet mask **255.255.255.224**. |

- **ECF:** if a wrong block size was found in 7(a), award points 1 and 2 for network and broadcast addresses that are correct for that block size.
- Accept a usable range of 172.16.45.193 to 172.16.45.222 as supporting working, but it does not replace the network and broadcast addresses.
- Writing /27 again does not earn point 3: the mask must be in dotted-decimal form.

## Question 8: DNS resolution [5 marks]

| Point | Marks | The candidate... |
|---|---|---|
| 1 | 1 | States the purpose: DNS translates a domain name into an IP address, using a distributed, hierarchical system of servers. |
| 2 | 1 | The browser / operating-system cache is checked first; on a miss the query goes to a recursive resolver (for example at the ISP). |
| 3 | 1 | The recursive resolver asks a root server, which refers it to the TLD server for .edu. |
| 4 | 1 | The TLD server refers it to the authoritative name server for northbridge.edu, which returns the IP address. |
| 5 | 1 | The answer is cached at each level for its time to live (TTL), so later lookups are faster. |

- **Keywords:** domain name, IP address, recursive resolver, root, TLD, .edu, authoritative, cache, TTL.
- **Order:** points 3 and 4 require the correct order (root → TLD → authoritative). If the servers are named but in the wrong order, award 1 of the 2 marks for points 3 and 4 together.
- **Accept** "DNS uses UDP port 53" as supporting detail, but it does not replace any point.
- A bare list "root, TLD, authoritative" earns at most half of points 3 and 4 (principle 4).

## Question 9: ACID [4 marks]

| Point | Marks | The candidate... |
|---|---|---|
| 1 | 1 | **Atomicity:** all of the transaction happens or none of it does; a failure part-way undoes (rolls back) the completed steps. |
| 2 | 1 | **Consistency:** a transaction takes the database from one valid state to another, respecting all constraints. |
| 3 | 1 | **Isolation:** concurrent transactions do not interfere; the result is as if they ran one after another. |
| 4 | 1 | **Durability:** once a transaction commits, its changes survive crashes and power failures. |

- **Partial credit:** the name of a property with no (or a wrong) explanation earns ½ for that point. A bare list of the four names therefore earns 2 marks at most.
- **Accept** a correct example in place of the explanation (for example, a bank transfer that is undone if the credit fails, for atomicity).
- **Do not accept** "consistency means the data is the same everywhere": that is a different (distributed-systems) meaning.

## Question 10(a): primary key vs foreign key [3 marks]

| Point | Marks | The candidate... |
|---|---|---|
| 1 | 1 | A primary key is the column (or minimal set of columns) chosen to uniquely identify each row of a table, and it cannot be NULL. |
| 2 | 1 | A foreign key is a column in one table that refers to the primary key of another table, linking the two tables (and enforcing referential integrity). |
| 3 | 1 | Gives a correct example showing both, for example student_id is the primary key of Students and a foreign key in Enrolments. |

- **Keywords:** uniquely identifies, NULL, refers to, another table, referential integrity.
- **Point 1:** "uniquely identifies each row" is required; "cannot be NULL" is not required for the mark.
- **Point 3:** the example must show the *link* between two tables. An example of a primary key alone earns ½.

## Question 10(b): normalising to 2NF [4 marks]

| Point | Marks | The candidate... |
|---|---|---|
| 1 | 1 | Identifies a partial dependency in this table: student_id → student_name, or course_code → course_title. |
| 2 | 1 | Explains why that breaks 2NF: a non-key column depends on only *part* of the composite primary key. |
| 3 | 1 | Gives a correct 2NF decomposition: Students(student_id, student_name), Courses(course_code, course_title) and Enrolments(student_id, course_code, grade). |
| 4 | 1 | Names and briefly describes one anomaly: update (a student's name stored in many rows), insertion (cannot add a course until a student enrols) or deletion (removing the last student in a course loses the course's title). |

- **Keywords:** partial dependency, composite key, part of the key, decompose, update anomaly, insertion anomaly, deletion anomaly.
- **Point 3:** the decomposition must keep grade with the full key (student_id, course_code). Two of the three tables correct earns ½.
- **Point 4:** naming the anomaly with no description earns ½.
- Do not award point 2 for "it has a transitive dependency": that is a 3NF problem, not 2NF.

---

## Summary of marks

| Question | Part | Marks | Type |
|---|---|---|---|
| 1 | | 3 | Short explanation |
| 2 | (a) | 2 | Definition |
| 2 | (b) | 3 | Compare |
| 3 | | 4 | Definition + recall |
| 4 | (a) | 2 | Definition |
| 4 | (b) | 4 | Explain |
| 5 | | 5 | Numerical |
| 6 | (a) | 3 | Explain a process |
| 6 | (b) | 3 | Compare |
| 7 | (a) | 2 | Numerical |
| 7 | (b) | 3 | Numerical |
| 8 | | 5 | Explain a process |
| 9 | | 4 | Definition |
| 10 | (a) | 3 | Compare |
| 10 | (b) | 4 | Explain / apply |
| **Total** | | **50** | |
