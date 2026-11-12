---
title: "CS201 Unit 1: Operating Systems"
subject: Operating Systems
unit: 1
---

# Unit 1: Operating Systems

**CS201 Systems Fundamentals** · Northbridge Institute, School of Computing
Course notes by Dr. Emma Carter · Edition 2026

## Learning outcomes

By the end of this unit you should be able to:

1. Explain what an operating system does and why it runs in a privileged mode.
2. Distinguish processes from threads and describe the life cycle of a process.
3. Compare CPU scheduling algorithms and calculate waiting and turnaround times.
4. Explain race conditions and use mutexes and semaphores to prevent them.
5. Define deadlock, state the four necessary conditions, and describe strategies for handling it.
6. Explain paging and virtual memory, and calculate page-table sizes.

## 1.1 What an operating system does

An **operating system (OS)** is the software layer that sits between applications and the hardware. It has two jobs that are easy to remember and harder to do well:

- **Resource manager.** The OS shares the CPU, memory, storage and devices fairly and efficiently among many running programs.
- **Abstraction provider.** The OS hides messy hardware details behind clean interfaces. A program writes to a *file*; it does not need to know which disk sectors are involved.

The core of the OS is the **kernel**. Modern processors run in at least two modes:

- **Kernel mode** (privileged): can execute any instruction and access any memory or device.
- **User mode** (restricted): ordinary applications run here and cannot touch hardware directly.

When an application needs a privileged service, such as reading a file, creating a process or sending network data, it makes a **system call**. A system call deliberately switches the CPU from user mode into kernel mode, the kernel performs the work on the program's behalf, and control returns to user mode. This separation protects the system: a bug in one application cannot corrupt the kernel or other programs.

## 1.2 Processes

A **process** is a program in execution. A program is a passive file on disk; a process is an active entity with its own state. Each process has:

- its own **address space**: code, data, heap and stack;
- a **program counter** indicating the next instruction;
- CPU **registers**;
- open files and other resources.

The OS keeps all of this in a data structure called the **Process Control Block (PCB)**: process ID, state, program counter, registers, scheduling priority, memory-management information and a list of open files.

### Process states

A process moves through five states during its life:

| State | Meaning |
|---|---|
| New | The process is being created. |
| Ready | It is waiting to be given the CPU. |
| Running | Its instructions are executing on a CPU. |
| Waiting (Blocked) | It is waiting for an event, such as I/O completing. |
| Terminated | It has finished execution. |

A running process moves to *waiting* when it requests I/O, back to *ready* when the I/O completes, and from *running* to *ready* when the scheduler pre-empts it (for example, when its time slice expires).

### Context switching

Switching the CPU from one process to another is called a **context switch**. The OS saves the state of the old process into its PCB and loads the saved state of the new process. Context switches are pure overhead (no useful user work happens during them), so a good OS keeps them fast and not too frequent.

## 1.3 Threads

A **thread** is the smallest unit of execution that the OS schedules. A process contains one or more threads. Threads of the same process **share** the process's code, data, heap and open files, but each thread has its **own** program counter, registers and stack.

| | Process | Thread |
|---|---|---|
| Memory | Separate address space | Shares the address space of its process |
| Creation cost | High (new address space) | Low |
| Context switch | Slower | Faster |
| Communication | Needs inter-process communication (pipes, sockets, shared memory) | Directly through shared variables |
| Isolation | A crash usually affects only that process | A crash in one thread can bring down the whole process |

Threads are useful for **responsiveness** (a user interface stays responsive while a background thread works), **resource sharing**, **economy** and **scalability** on multi-core processors. The price of sharing is that threads can interfere with each other's data, which leads directly to the synchronisation problems in section 1.5.

## 1.4 CPU scheduling

The **CPU scheduler** decides which ready process runs next. Scheduling can be **non-preemptive** (a process keeps the CPU until it blocks or finishes) or **preemptive** (the OS can take the CPU away, for example when a time slice expires).

We judge schedulers with these metrics:

- **Turnaround time** = completion time − arrival time
- **Waiting time** = turnaround time − CPU burst time
- **Response time** = time of first run − arrival time
- **Throughput** = number of processes completed per unit of time

### Common algorithms

- **First-Come, First-Served (FCFS):** run processes in arrival order. Simple, but a long job at the front makes every short job wait: the **convoy effect**.
- **Shortest Job First (SJF):** run the process with the shortest next CPU burst. SJF gives the minimum average waiting time, but it needs burst lengths in advance and can **starve** long jobs.
- **Round Robin (RR):** each process gets a fixed **time quantum**; if it hasn't finished, it goes to the back of the ready queue. RR is fair and gives good response times. A very large quantum behaves like FCFS; a very small quantum wastes time on context switches.
- **Priority scheduling:** run the highest-priority process. Low-priority processes may starve; **ageing** (gradually raising the priority of waiting processes) fixes this.

### Worked example: FCFS vs SJF

Three processes arrive at time 0 with CPU bursts P1 = 24 ms, P2 = 3 ms and P3 = 3 ms.

- **FCFS (order P1, P2, P3):** waiting times are P1 = 0, P2 = 24, P3 = 27. Average waiting time = (0 + 24 + 27) / 3 = **17 ms**.
- **SJF (order P2, P3, P1):** waiting times are P2 = 0, P3 = 3, P1 = 6. Average waiting time = (0 + 3 + 6) / 3 = **3 ms**.

The same work, scheduled differently, cuts the average wait from 17 ms to 3 ms. This is the convoy effect in numbers.

## 1.5 Synchronisation

When two threads access shared data at the same time and the result depends on the exact order of their steps, we have a **race condition**. Example: two threads each execute `counter = counter + 1`. Each reads the old value, adds one and writes back. If both read before either writes, one increment is lost.

The section of code that accesses shared data is the **critical section**. A correct solution to the critical-section problem must satisfy three requirements:

1. **Mutual exclusion:** at most one thread is in the critical section at a time.
2. **Progress:** if no thread is in the critical section, the choice of who enters next cannot be postponed forever.
3. **Bounded waiting:** there is a limit on how many times other threads can enter before a waiting thread gets its turn.

### Mutex locks

A **mutex** (mutual-exclusion lock) is the simplest tool. A thread calls `acquire()` before the critical section and `release()` after it. Only one thread can hold the mutex at a time.

### Semaphores

A **semaphore** is an integer variable that, apart from initialisation, is accessed only through two **atomic** operations:

- **wait()** (also called **P** or down): decrement the semaphore; if the value becomes negative, the calling thread blocks.
- **signal()** (also called **V** or up): increment the semaphore; if threads are blocked, wake one of them.

*Atomic* means each operation happens as a single indivisible step: no other thread can see it half-done. A **binary semaphore** takes only the values 0 and 1 and behaves like a mutex. A **counting semaphore** can take any non-negative value and is used to control access to a resource with several identical instances. For example, a semaphore initialised to 5 allows at most five threads to use a pool of five database connections at once.

The classic **producer–consumer (bounded-buffer)** problem uses three semaphores: `mutex` (initially 1) protects the buffer, `empty` (initially N) counts free slots, and `full` (initially 0) counts filled slots.

### Common misconceptions

- *"A semaphore is just a lock."* Only a binary semaphore behaves like a lock; a counting semaphore allows several threads in at once.
- *"Race conditions only happen on multi-core machines."* They can happen on a single core too, because the OS can switch threads between any two instructions.

## 1.6 Deadlocks

A **deadlock** is a situation in which a set of processes are all blocked, and each process is waiting for a resource that is held by another process in the same set. None of them can ever proceed.

### The four necessary conditions (Coffman conditions)

A deadlock can arise only if **all four** of these conditions hold at the same time:

1. **Mutual exclusion:** at least one resource is non-shareable; only one process can use it at a time.
2. **Hold and wait:** a process holds at least one resource while waiting to acquire others.
3. **No preemption:** a resource cannot be taken away from a process; it is released only voluntarily.
4. **Circular wait:** there is a cycle of processes P1 → P2 → … → Pn → P1, where each process waits for a resource held by the next.

Because all four are *necessary*, breaking any one of them prevents deadlock.

### Resource-allocation graphs

We can draw processes and resources as a graph: an edge from a process to a resource means "requests", and an edge from a resource to a process means "is held by". If every resource has a single instance, **a cycle in the graph means deadlock**. With multiple instances, a cycle means deadlock is *possible* but not certain.

### Handling deadlocks

| Strategy | Idea | Example |
|---|---|---|
| Prevention | Make sure at least one of the four conditions can never hold | Impose a global ordering on resources so that circular wait is impossible |
| Avoidance | Grant a request only if the system stays in a *safe state* | The **Banker's algorithm** |
| Detection and recovery | Let deadlocks happen, detect cycles, then recover | Abort a process, or pre-empt its resources |
| Ignore it | Assume deadlocks are rare and restart if needed | The "ostrich algorithm" used by many general-purpose OSes |

### Deadlock vs starvation

These are often confused. In a **deadlock**, the processes involved can *never* proceed. In **starvation**, a process *could* proceed but keeps being overlooked, for example a low-priority job that is always passed over. Starvation is solved by ageing; deadlock requires one of the strategies above.

## 1.7 Memory management

Programs use **logical (virtual) addresses**; the hardware uses **physical addresses**. The **Memory Management Unit (MMU)** translates one into the other on every memory access.

### Fragmentation

- **External fragmentation:** there is enough free memory in total, but it is split into small non-contiguous holes, so a large request cannot be satisfied.
- **Internal fragmentation:** memory allocated to a process is slightly larger than it asked for, and the unused part inside the allocation is wasted.

### Paging

**Paging** removes external fragmentation. Logical memory is divided into fixed-size blocks called **pages**, and physical memory into blocks of the same size called **frames**. Any page can go into any free frame. A **page table** for each process records which frame holds each page.

A logical address is split into a **page number** (used as an index into the page table) and an **offset** (the position within the page). With a page size of 2^d bytes, the low d bits are the offset.

### Worked example: page-table size

A system has **32-bit logical addresses**, a **page size of 4 KB** and **4-byte page-table entries**.

1. Page size = 4 KB = 2^12 bytes, so the offset uses **12 bits**.
2. The page number uses 32 − 12 = **20 bits**, so there are 2^20 = **1,048,576 pages**.
3. One entry per page: page-table size = 2^20 × 4 bytes = 2^22 bytes = **4 MB per process**.

A 4 MB page table for *every* process is expensive, which is why real systems use **multi-level (hierarchical) page tables** that only allocate the parts of the table actually in use.

### The TLB

Every memory access would need *two* memory accesses: one for the page table and one for the data. The **Translation Lookaside Buffer (TLB)** is a small, fast hardware cache of recent page-to-frame translations. Most lookups hit the TLB, so translation is usually almost free.

### Virtual memory

**Virtual memory** lets a process use more memory than is physically available by keeping only the pages it is currently using in RAM. With **demand paging**, a page is loaded only when first accessed. Accessing a page that is not in memory causes a **page fault**: the OS finds the page on disk, loads it into a frame (evicting another page if necessary) and restarts the instruction.

When memory is full, a **page-replacement algorithm** chooses the victim:

- **FIFO:** evict the oldest page. Simple, but can suffer **Belady's anomaly** (more frames can cause more faults).
- **Optimal (OPT):** evict the page that will not be used for the longest time. Impossible in practice (it needs the future), but useful as a benchmark.
- **LRU (Least Recently Used):** evict the page unused for the longest time. A good, practical approximation of OPT.

If processes do not have enough frames for their working sets, the system spends more time paging than executing. This collapse in performance is called **thrashing**.

## 1.8 File systems in brief

A **file** is a named collection of related data stored on secondary storage. A **directory** maps names to files. Files can be allocated on disk in three main ways: **contiguous** (fast, but suffers external fragmentation), **linked** (no fragmentation, but slow random access) and **indexed** (an index block lists every block of the file, supporting fast random access).

## Key terms

| Term | Definition |
|---|---|
| Kernel | The core of the OS, running in privileged mode |
| System call | A controlled entry point from a user program into the kernel |
| Process | A program in execution, with its own address space |
| Thread | A unit of execution within a process; threads share the process's memory |
| Context switch | Saving one process's state and loading another's |
| Race condition | An outcome that depends on the timing of concurrent accesses to shared data |
| Critical section | Code that accesses shared data and must not run concurrently |
| Semaphore | An integer accessed only through atomic wait() and signal() operations |
| Deadlock | A set of processes each waiting for a resource held by another in the set |
| Starvation | A process waits indefinitely because others are always chosen first |
| Paging | Dividing memory into fixed-size pages and frames |
| Page fault | An access to a page that is not currently in physical memory |
| Thrashing | Excessive paging that leaves little time for useful work |

## Unit summary

- The OS manages resources and provides abstractions; the kernel runs in privileged mode and is entered through system calls.
- A process has its own address space; threads within a process share memory but have their own stacks and registers.
- Scheduling choices matter: SJF minimises average waiting time, Round Robin gives fairness and good response.
- Race conditions are prevented by protecting critical sections with mutexes or semaphores.
- Deadlock needs all four conditions (mutual exclusion, hold and wait, no preemption, circular wait); breaking any one prevents it.
- Paging removes external fragmentation; virtual memory and demand paging let programs exceed physical memory, at the risk of thrashing.
