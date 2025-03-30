#	计科2302 吴智明
#	数据库实验一

# 1. 创建数据库studentInfo，包含如下表
create database studentInfo;
use studentInfo;

# 1.1 student表
create table student(
	Student_id varchar(10) not null,
    Student_name varchar(10) not null,
    sex varchar(1) not null check(sex in ('F','M')),
    age int,
    department varchar(15) default 'computer',
    primary key(Student_id)
    );

# 1.2 course表
create table course(
	Course_id varchar(6) not null,
    Course_name varchar(20) not null,
    PreCouId varchar(6),
    Credits Decimal(3,1) not null,
    primary key(Course_id)
    );

# 1.3 score表    
create table score(
	Student_id varchar(10) not null,
    Course_id varchar(20) not null,
    Grade Decimal(3,1) check(Grade > 0 and Grade < 100 ),
    primary key(Student_id , Course_id),
    foreign key(Student_id) references student(Student_id),
    foreign key(Course_id) references course(Course_id)
    );
   
# 2. 增加、修改、删除字段

# 2.1 为表student增加一个memo（备注）字段，类型为varchar（200）
alter table student 
add column memo varchar(200) null;

# 2.2 将memo字段的数据类型更改为varchar（300）
alter table student
modify column memo varchar(300) ;

# 2.3 删除memo字段
alter table student
drop column memo;
 
# 3 删除表
Drop table student,course,score;





    