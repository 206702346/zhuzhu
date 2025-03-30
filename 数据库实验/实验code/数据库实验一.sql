#	计科2302 吴智明
#	数据库实验一
 
#3 创建数据库
CREATE DATABASE Test2
on primary(
	name=Test2_data1,
	filename='/Users/mac/Documents/sql/Test2_data1.mdf',
	size=10,
	maxsize=100,
	filegrowth=10
	),
	
	(
	name=Test2_data2,
	filename='/Users/mac/Documents/sql/Test2_data2.ndf'
	size=10,
	maxsize=10,
	filegrowth=15%
	)
	
	log on(
	name=Test2_log1,
	filename='/Users/mac/Documents/sql/Test2_log1.ldf'
	size=10,
	maxsize=50,
	filegrowth=20
	)
	
#4 修改数据库
alter database Test2
modify file (
	name = Test2_data1,
    size=50,
    maxsize=200,
    filegrowth=20
)

alter database Test2
modify file (
	name = Test2_data2,
    size=50,
    maxsize=300,
    filegrowth=20
)

alter database Test2
modify file (
	name = Test2_log1,
    size=30,
    maxsize=100,
    filegrowth=10%
)

#5 数据库更名
alter database new_Test1
modify name = Test1

#6 删除数据库
drop database Test2

	
