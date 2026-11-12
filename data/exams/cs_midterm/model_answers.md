# CS201 Midterm: Model Answers

**Northbridge Institute, School of Computing** · CS201 Systems Fundamentals · Exam code CS201-MT-2026 · 50 marks

> 💡 These are full-marks answers written by the examiner, Dr. Emma Carter. They are deliberately a little longer than a student needs to write in 90 minutes: a shorter answer that hits every point of the marking scheme earns the same marks. Each section heading is a question ID used by GradeWise (`Q7a` = Question 7(a)).

## Q1

**Question 1. What is a system call? Explain why the CPU switches from user mode to kernel mode when a system call is made. [3 marks]**

A system call is a controlled request from a user program to the operating system kernel for a privileged service, such as reading a file, creating a process or sending data over the network. Ordinary applications run in user mode, which is restricted: they cannot execute privileged instructions or access hardware directly. The kernel runs in kernel mode, which is privileged. When a system call is made, the CPU deliberately switches into kernel mode so that the kernel can perform the work on the program's behalf, and then it returns to user mode. This separation protects the system: a bug in one application cannot corrupt the kernel or other programs.

## Q2a

**Question 2(a). Define a process. [2 marks]**

A process is a program in execution. A program is a passive file on disk; a process is an active entity with its own state: its own address space (code, data, heap and stack), a program counter, CPU registers and open files, all recorded by the OS in its Process Control Block (PCB).

## Q2b

**Question 2(b). Give THREE differences between a process and a thread. [3 marks]**

1. **Memory:** each process has its own separate address space, whereas the threads of a process share its code, data, heap and open files (each thread has only its own stack, registers and program counter).
2. **Cost:** creating a process and switching between processes is expensive, because a new address space is involved; creating and switching threads is much cheaper and faster.
3. **Communication and isolation:** processes must use inter-process communication (pipes, sockets, shared memory) to talk to each other, and a crash usually affects only one process; threads communicate directly through shared variables, but a crash in one thread can bring down the whole process.

## Q3

**Question 3. What is a deadlock? Name TWO of the four necessary conditions for a deadlock. [4 marks]**

A deadlock is a situation in which a set of processes are all blocked, and each process is waiting for a resource that is held by another process in the same set, so none of them can ever proceed.

Two of the four necessary (Coffman) conditions are:
1. **Mutual exclusion:** at least one resource is non-shareable, so only one process can use it at a time.
2. **Circular wait:** there is a cycle of processes P1 → P2 → … → Pn → P1 in which each waits for a resource held by the next.

(The other two are hold and wait, and no preemption.)

## Q4a

**Question 4(a). What is a race condition? [2 marks]**

A race condition occurs when two or more threads or processes access shared data at the same time, at least one of them writing to it, and the final result depends on the exact timing or order in which their steps are interleaved. For example, if two threads both run `counter = counter + 1` and both read the old value before either writes back, one increment is lost.

## Q4b

**Question 4(b). Explain how a semaphore is used for synchronisation, describing wait() and signal() and the difference between a binary and a counting semaphore. [4 marks]**

A semaphore is an integer variable that, apart from being initialised, is accessed only through two atomic (indivisible) operations:

- **wait()** (also called P or down) decrements the semaphore. If the value becomes negative, the calling thread blocks until it is woken.
- **signal()** (also called V or up) increments the semaphore and, if any threads are blocked on it, wakes one of them.

A thread calls wait() before entering the critical section and signal() after leaving it. A **binary semaphore** takes only the values 0 and 1 and behaves like a mutex lock, allowing one thread in at a time. A **counting semaphore** can take any non-negative value and controls access to a resource with several identical instances: a semaphore initialised to 5 lets at most five threads use a pool of five database connections at once.

## Q5

**Question 5. 32-bit logical addresses, 8 KB pages, 4-byte page-table entries: offset bits, number of pages, page-table size. [5 marks]**

(i) Page size = 8 KB = 8 × 1024 = 2^13 bytes, so the offset uses **13 bits**.

(ii) Page-number bits = 32 − 13 = 19 bits, so the number of pages = 2^19 = **524,288 pages**.

(iii) There is one entry per page, so page-table size = number of pages × entry size = 2^19 × 4 bytes = 2^19 × 2^2 = 2^21 bytes = **2 MB per process**.

## Q6a

**Question 6(a). Describe the TCP three-way handshake. [3 marks]**

1. **SYN:** the client sends a segment with the SYN flag set, carrying its initial sequence number.
2. **SYN-ACK:** the server replies with a segment that has both the SYN and ACK flags set: it acknowledges the client's sequence number and sends its own initial sequence number.
3. **ACK:** the client sends an ACK acknowledging the server's sequence number. The connection is now established and data can flow in both directions.

## Q6b

**Question 6(b). Give TWO differences between TCP and UDP, and name ONE application better suited to UDP, explaining why. [3 marks]**

1. TCP is connection-oriented (it opens a connection with a handshake first), whereas UDP is connectionless.
2. TCP is reliable and ordered: lost segments are retransmitted and bytes are delivered in order. UDP gives no retransmission and no ordering guarantee, but has lower overhead (an 8-byte header instead of at least 20 bytes).

A **video call** is better suited to UDP: it is real-time, so a late packet is useless, and waiting for TCP's retransmissions would add delay. Low latency matters more than perfect reliability, and the application can cope with an occasional lost packet.

## Q7a

**Question 7(a). 172.16.45.200/27: how many usable host addresses? [2 marks]**

Host bits = 32 − 27 = 5, so the subnet has 2^5 = 32 addresses. Two are reserved (the network address and the broadcast address), so there are 32 − 2 = **30 usable host addresses**.

## Q7b

**Question 7(b). Network address, broadcast address and subnet mask of 172.16.45.200/27. [3 marks]**

The block size is 32, so subnets start at multiples of 32 in the last octet: 0, 32, 64, 96, 128, 160, 192, 224. Since 200 lies between 192 and 223:

- Network address = **172.16.45.192**
- Broadcast address = **172.16.45.223**
- Subnet mask for /27 = 27 one-bits = **255.255.255.224** (256 − 32 = 224)

(Usable host range: 172.16.45.193 to 172.16.45.222.)

## Q8

**Question 8. Explain, step by step, how DNS resolves www.northbridge.edu. [5 marks]**

DNS (the Domain Name System) is a distributed, hierarchical database that translates domain names such as www.northbridge.edu into the IP addresses that IP needs.

1. The browser checks its own cache, then asks the operating system's resolver. Because the site has never been visited, there is no cached answer, so the query goes to a **recursive resolver**, usually run by the ISP.
2. The recursive resolver asks a **root server**, which does not know the address but refers it to the **top-level domain (TLD) server for .edu**.
3. The .edu TLD server refers the resolver to the **authoritative name server for northbridge.edu**.
4. The authoritative server returns the IP address of www.northbridge.edu. The recursive resolver passes it back to the computer, and the browser can now connect.
5. The answer is **cached** at each level for its **time to live (TTL)**, so the next lookup is much faster.

DNS queries normally travel over UDP on port 53, because they are small and need to be fast.

## Q9

**Question 9. State and briefly explain the four ACID properties. [4 marks]**

- **Atomicity:** all of a transaction happens, or none of it does. If a bank transfer fails after the debit but before the credit, the debit is undone (rolled back).
- **Consistency:** a transaction takes the database from one valid state to another valid state, respecting every integrity constraint.
- **Isolation:** concurrent transactions do not interfere with each other; the result is as if they had run one after another.
- **Durability:** once a transaction commits, its changes are permanent and survive crashes and power failures.

## Q10a

**Question 10(a). Explain the difference between a primary key and a foreign key, using an example. [3 marks]**

A **primary key** is the column (or minimal set of columns) chosen to uniquely identify each row of a table; it can never be NULL. A **foreign key** is a column in one table that refers to the primary key of another table, linking the two tables; referential integrity requires every foreign-key value to match an existing row in the referenced table.

Example: in `Students(student_id, name, programme)`, `student_id` is the primary key. In `Enrolments(student_id, course_code, grade)`, `student_id` is a foreign key referring to `Students`, so you cannot enrol a student who does not exist.

## Q10b

**Question 10(b). Enrolment(student_id, course_code, student_name, course_title, grade) with key (student_id, course_code): why not 2NF, decompose, name one anomaly. [4 marks]**

The table has partial dependencies: `student_id → student_name` and `course_code → course_title`. Second normal form forbids any non-key column depending on only *part* of a composite primary key, and student_name and course_title each depend on just one half of (student_id, course_code), so the table is not in 2NF.

Decomposition into 2NF:
- `Students(student_id, student_name)`
- `Courses(course_code, course_title)`
- `Enrolments(student_id, course_code, grade)`, with primary key (student_id, course_code); grade depends on the whole key.

One anomaly: an **update anomaly**. A student's name is repeated in every row for every course they take, so changing it means updating many rows; missing one leaves the data inconsistent. (Also accepted: an insertion anomaly, since a new course cannot be added until a student enrols in it, or a deletion anomaly, since removing the last student in a course loses its title.)
