-- SQLite
-- Data definition 'elem'
CREATE TABLE users(
    id INTEGER NOT NULL PRIMARY KEY,
    email text not null,
    password not NULL
);
INSERT INTO users(email,password) VALUES('achraf@ensam.com','0000');
INSERT INTO users(email,password) VALUES('achraf1@ensam.com','0001');
INSERT INTO users(email,password) VALUES('achraf2@ensam.com','0002');
INSERT INTO users(email,password) VALUES('achraf3@ensam.com','0003');
INSERT INTO users(email,password) VALUES('achraf4@ensam.com','0004');
INSERT INTO users(email,password) VALUES('achraf5@ensam.com','0005');
INSERT INTO users(email,password) VALUES('achraf6@ensam.com','0006');
INSERT INTO users(email,password) VALUES('achraf7@ensam.com','0007');
INSERT INTO users(email,password) VALUES('achraf8@ensam.com','0008');
INSERT INTO users(email,password) VALUES('achraf90@ensam.com','0009');
-- Data manipulation 'elements.db'
SELECT * FROM users;

DROP table users**;


-- ************************************ Track *************************************
-- DDF
CREATE TABLE [Track]
(
    [TrackId] INTEGER  NOT NULL,
    [Name] NVARCHAR(200)  NOT NULL,
    [AlbumId] INTEGER,
    [MediaTypeId] INTEGER  NOT NULL,
    [GenreId] INTEGER,
    [Composer] NVARCHAR(220),
    [Milliseconds] INTEGER  NOT NULL,
    [Bytes] INTEGER,
    [UnitPrice] NUMERIC(10,2)  NOT NULL,
    CONSTRAINT [PK_Track] PRIMARY KEY  ([TrackId])
);
-- INSERT
INSERT INTO [Track] ([TrackId], [Name], [AlbumId], [MediaTypeId], [GenreId], [Composer], [Milliseconds], [Bytes], [UnitPrice]) VALUES (116, 'C''Mon Everybody', 12, 1, 5, 'Eddie Cochran/Jerry Capehart', 140199, 2247846, 0.99);
INSERT INTO [Track] ([TrackId], [Name], [AlbumId], [MediaTypeId], [GenreId], [Composer], [Milliseconds], [Bytes], [UnitPrice]) VALUES (117, 'Rock ''N'' Roll Music', 12, 1, 5, 'Chuck Berry', 141923, 2276788, 0.99);
INSERT INTO [Track] ([TrackId], [Name], [AlbumId], [MediaTypeId], [GenreId], [Composer], [Milliseconds], [Bytes], [UnitPrice]) VALUES (118, 'Slow Down', 12, 1, 5, 'Larry Williams', 163265, 2616981, 0.99);
INSERT INTO [Track] ([TrackId], [Name], [AlbumId], [MediaTypeId], [GenreId], [Composer], [Milliseconds], [Bytes], [UnitPrice]) VALUES (119, 'Roadrunner', 12, 1, 5, 'Bo Diddley', 143595, 2301989, 0.99);
INSERT INTO [Track] ([TrackId], [Name], [AlbumId], [MediaTypeId], [GenreId], [Composer], [Milliseconds], [Bytes], [UnitPrice]) VALUES (120, 'Carol', 12, 1, 5, 'Chuck Berry', 143830, 2306019, 0.99);
INSERT INTO [Track] ([TrackId], [Name], [AlbumId], [MediaTypeId], [GenreId], [Composer], [Milliseconds], [Bytes], [UnitPrice]) VALUES (121, 'Good Golly Miss Molly', 12, 1, 5, 'Little Richard', 106266, 1704918, 0.99);
INSERT INTO [Track] ([TrackId], [Name], [AlbumId], [MediaTypeId], [GenreId], [Composer], [Milliseconds], [Bytes], [UnitPrice]) VALUES (122, '20 Flight Rock', 12, 1, 5, 'Ned Fairchild', 107807, 1299960, 0.99);
INSERT INTO [Track] ([TrackId], [Name], [AlbumId], [MediaTypeId], [GenreId], [Composer], [Milliseconds], [Bytes], [UnitPrice]) VALUES (123, 'Quadrant', 13, 1, 2, 'Billy Cobham', 261851, 8538199, 0.99);
INSERT INTO [Track] ([TrackId], [Name], [AlbumId], [MediaTypeId], [GenreId], [Composer], [Milliseconds], [Bytes], [UnitPrice]) VALUES (124, 'Snoopy''s search-Red baron', 13, 1, 2, 'Billy Cobham', 456071, 15075616, 0.99);
INSERT INTO [Track] ([TrackId], [Name], [AlbumId], [MediaTypeId], [GenreId], [Composer], [Milliseconds], [Bytes], [UnitPrice]) VALUES (125, 'Spanish moss-"A sound portrait"-Spanish moss', 13, 1, 2, 'Billy Cobham', 248084, 8217867, 0.99);
INSERT INTO [Track] ([TrackId], [Name], [AlbumId], [MediaTypeId], [GenreId], [Composer], [Milliseconds], [Bytes], [UnitPrice]) VALUES (126, 'Moon germs', 13, 1, 2, 'Billy Cobham', 294060, 9714812, 0.99);
INSERT INTO [Track] ([TrackId], [Name], [AlbumId], [MediaTypeId], [GenreId], [Composer], [Milliseconds], [Bytes], [UnitPrice]) VALUES (127, 'Stratus', 13, 1, 2, 'Billy Cobham', 582086, 19115680, 0.99);
INSERT INTO [Track] ([TrackId], [Name], [AlbumId], [MediaTypeId], [GenreId], [Composer], [Milliseconds], [Bytes], [UnitPrice]) VALUES (128, 'The pleasant pheasant', 13, 1, 2, 'Billy Cobham', 318066, 10630578, 0.99);
INSERT INTO [Track] ([TrackId], [Name], [AlbumId], [MediaTypeId], [GenreId], [Composer], [Milliseconds], [Bytes], [UnitPrice]) VALUES (129, 'Solo-Panhandler', 13, 1, 2, 'Billy Cobham', 246151, 8230661, 0.99);
INSERT INTO [Track] ([TrackId], [Name], [AlbumId], [MediaTypeId], [GenreId], [Composer], [Milliseconds], [Bytes], [UnitPrice]) VALUES (130, 'Do what cha wanna', 13, 1, 2, 'George Duke', 274155, 9018565, 0.99);
INSERT INTO [Track] ([TrackId], [Name], [AlbumId], [MediaTypeId], [GenreId], [Milliseconds], [Bytes], [UnitPrice]) VALUES (131, 'Intro/ Low Down', 14, 1, 3, 323683, 10642901, 0.99);
INSERT INTO [Track] ([TrackId], [Name], [AlbumId], [MediaTypeId], [GenreId], [Milliseconds], [Bytes], [UnitPrice]) VALUES (132, '13 Years Of Grief', 14, 1, 3, 246987, 8137421, 0.99);
INSERT INTO [Track] ([TrackId], [Name], [AlbumId], [MediaTypeId], [GenreId], [Milliseconds], [Bytes], [UnitPrice]) VALUES (133, 'Stronger Than Death', 14, 1, 3, 300747, 9869647, 0.99);
INSERT INTO [Track] ([TrackId], [Name], [AlbumId], [MediaTypeId], [GenreId], [Milliseconds], [Bytes], [UnitPrice]) VALUES (134, 'All For You', 14, 1, 3, 235833, 7726948, 0.99);
INSERT INTO [Track] ([TrackId], [Name], [AlbumId], [MediaTypeId], [GenreId], [Milliseconds], [Bytes], [UnitPrice]) VALUES (135, 'Super Terrorizer', 14, 1, 3, 319373, 10513905, 0.99);
INSERT INTO [Track] ([TrackId], [Name], [AlbumId], [MediaTypeId], [GenreId], [Milliseconds], [Bytes], [UnitPrice]) VALUES (136, 'Phoney Smile Fake Hellos', 14, 1, 3, 273606, 9011701, 0.99);
INSERT INTO [Track] ([TrackId], [Name], [AlbumId], [MediaTypeId], [GenreId], [Milliseconds], [Bytes], [UnitPrice]) VALUES (137, 'Lost My Better Half', 14, 1, 3, 284081, 9355309, 0.99);
INSERT INTO [Track] ([TrackId], [Name], [AlbumId], [MediaTypeId], [GenreId], [Milliseconds], [Bytes], [UnitPrice]) VALUES (138, 'Bored To Tears', 14, 1, 3, 247327, 8130090, 0.99);
INSERT INTO [Track] ([TrackId], [Name], [AlbumId], [MediaTypeId], [GenreId], [Milliseconds], [Bytes], [UnitPrice]) VALUES (139, 'A.N.D.R.O.T.A.Z.', 14, 1, 3, 266266, 8574746, 0.99);
INSERT INTO [Track] ([TrackId], [Name], [AlbumId], [MediaTypeId], [GenreId], [Milliseconds], [Bytes], [UnitPrice]) VALUES (140, 'Born To Booze', 14, 1, 3, 282122, 9257358, 0.99);
INSERT INTO [Track] ([TrackId], [Name], [AlbumId], [MediaTypeId], [GenreId], [Milliseconds], [Bytes], [UnitPrice]) VALUES (141, 'World Of Trouble', 14, 1, 3, 359157, 11820932, 0.99);
INSERT INTO [Track] ([TrackId], [Name], [AlbumId], [MediaTypeId], [GenreId], [Milliseconds], [Bytes], [UnitPrice]) VALUES (142, 'No More Tears', 14, 1, 3, 555075, 18041629, 0.99);
INSERT INTO [Track] ([TrackId], [Name], [AlbumId], [MediaTypeId], [GenreId], [Milliseconds], [Bytes], [UnitPrice]) VALUES (143, 'The Begining... At Last', 14, 1, 3, 365662, 11965109, 0.99);
INSERT INTO [Track] ([TrackId], [Name], [AlbumId], [MediaTypeId], [GenreId], [Milliseconds], [Bytes], [UnitPrice]) VALUES (144, 'Heart Of Gold', 15, 1, 3, 194873, 6417460, 0.99);
INSERT INTO [Track] ([TrackId], [Name], [AlbumId], [MediaTypeId], [GenreId], [Milliseconds], [Bytes], [UnitPrice]) VALUES (145, 'Snowblind', 15, 1, 3, 420022, 13842549, 0.99);
INSERT INTO [Track] ([TrackId], [Name], [AlbumId], [MediaTypeId], [GenreId], [Milliseconds], [Bytes], [UnitPrice]) VALUES (146, 'Like A Bird', 15, 1, 3, 276532, 9115657, 0.99);
INSERT INTO [Track] ([TrackId], [Name], [AlbumId], [MediaTypeId], [GenreId], [Milliseconds], [Bytes], [UnitPrice]) VALUES (147, 'Blood In The Wall', 15, 1, 3, 284368, 9359475, 0.99);
INSERT INTO [Track] ([TrackId], [Name], [AlbumId], [MediaTypeId], [GenreId], [Milliseconds], [Bytes], [UnitPrice]) VALUES (148, 'The Beginning...At Last', 15, 1, 3, 271960, 8975814, 0.99);
INSERT INTO [Track] ([TrackId], [Name], [AlbumId], [MediaTypeId], [GenreId], [Milliseconds], [Bytes], [UnitPrice]) VALUES (149, 'Black Sabbath', 16, 1, 3, 382066, 12440200, 0.99);
INSERT INTO [Track] ([TrackId], [Name], [AlbumId], [MediaTypeId], [GenreId], [Milliseconds], [Bytes], [UnitPrice]) VALUES (150, 'The Wizard', 16, 1, 3, 264829, 8646737, 0.99);
INSERT INTO [Track] ([TrackId], [Name], [AlbumId], [MediaTypeId], [GenreId], [Milliseconds], [Bytes], [UnitPrice]) VALUES (151, 'Behind The Wall Of Sleep', 16, 1, 3, 217573, 7169049, 0.99);
INSERT INTO [Track] ([TrackId], [Name], [AlbumId], [MediaTypeId], [GenreId], [Milliseconds], [Bytes], [UnitPrice]) VALUES (152, 'N.I.B.', 16, 1, 3, 368770, 12029390, 0.99);
INSERT INTO [Track] ([TrackId], [Name], [AlbumId], [MediaTypeId], [GenreId], [Milliseconds], [Bytes], [UnitPrice]) VALUES (153, 'Evil Woman', 16, 1, 3, 204930, 6655170, 0.99);
INSERT INTO [Track] ([TrackId], [Name], [AlbumId], [MediaTypeId], [GenreId], [Milliseconds], [Bytes], [UnitPrice]) VALUES (154, 'Sleeping Village', 16, 1, 3, 644571, 21128525, 0.99);
INSERT INTO [Track] ([TrackId], [Name], [AlbumId], [MediaTypeId], [GenreId], [Milliseconds], [Bytes], [UnitPrice]) VALUES (155, 'Warning', 16, 1, 3, 212062, 6893363, 0.99);
INSERT INTO [Track] ([TrackId], [Name], [AlbumId], [MediaTypeId], [GenreId], [Composer], [Milliseconds], [Bytes], [UnitPrice]) VALUES (156, 'Wheels Of Confusion / The Straightener', 17, 1, 3, 'Tony Iommi, Bill Ward, Geezer Butler, Ozzy Osbourne', 494524, 16065830, 0.99);
INSERT INTO [Track] ([TrackId], [Name], [AlbumId], [MediaTypeId], [GenreId], [Composer], [Milliseconds], [Bytes], [UnitPrice]) VALUES (157, 'Tomorrow''s Dream', 17, 1, 3, 'Tony Iommi, Bill Ward, Geezer Butler, Ozzy Osbourne', 192496, 6252071, 0.99);
INSERT INTO [Track] ([TrackId], [Name], [AlbumId], [MediaTypeId], [GenreId], [Composer], [Milliseconds], [Bytes], [UnitPrice]) VALUES (158, 'Changes', 17, 1, 3, 'Tony Iommi, Bill Ward, Geezer Butler, Ozzy Osbourne', 286275, 9175517, 0.99);
INSERT INTO [Track] ([TrackId], [Name], [AlbumId], [MediaTypeId], [GenreId], [Composer], [Milliseconds], [Bytes], [UnitPrice]) VALUES (159, 'FX', 17, 1, 3, 'Tony Iommi, Bill Ward, Geezer Butler, Ozzy Osbourne', 103157, 3331776, 0.99);
INSERT INTO [Track] ([TrackId], [Name], [AlbumId], [MediaTypeId], [GenreId], [Composer], [Milliseconds], [Bytes], [UnitPrice]) VALUES (160, 'Supernaut', 17, 1, 3, 'Tony Iommi, Bill Ward, Geezer Butler, Ozzy Osbourne', 285779, 9245971, 0.99);
INSERT INTO [Track] ([TrackId], [Name], [AlbumId], [MediaTypeId], [GenreId], [Composer], [Milliseconds], [Bytes], [UnitPrice]) VALUES (161, 'Snowblind', 17, 1, 3, 'Tony Iommi, Bill Ward, Geezer Butler, Ozzy Osbourne', 331676, 10813386, 0.99);
INSERT INTO [Track] ([TrackId], [Name], [AlbumId], [MediaTypeId], [GenreId], [Composer], [Milliseconds], [Bytes], [UnitPrice]) VALUES (162, 'Cornucopia', 17, 1, 3, 'Tony Iommi, Bill Ward, Geezer Butler, Ozzy Osbourne', 234814, 7653880, 0.99);
INSERT INTO [Track] ([TrackId], [Name], [AlbumId], [MediaTypeId], [GenreId], [Composer], [Milliseconds], [Bytes], [UnitPrice]) VALUES (163, 'Laguna Sunrise', 17, 1, 3, 'Tony Iommi, Bill Ward, Geezer Butler, Ozzy Osbourne', 173087, 5671374, 0.99);
INSERT INTO [Track] ([TrackId], [Name], [AlbumId], [MediaTypeId], [GenreId], [Composer], [Milliseconds], [Bytes], [UnitPrice]) VALUES (164, 'St. Vitus Dance', 17, 1, 3, 'Tony Iommi, Bill Ward, Geezer Butler, Ozzy Osbourne', 149655, 4884969, 0.99);
INSERT INTO [Track] ([TrackId], [Name], [AlbumId], [MediaTypeId], [GenreId], [Composer], [Milliseconds], [Bytes], [UnitPrice]) VALUES (165, 'Under The Sun/Every Day Comes and Goes', 17, 1, 3, 'Tony Iommi, Bill Ward, Geezer Butler, Ozzy Osbourne', 350458, 11360486, 0.99);