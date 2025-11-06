-- phpMyAdmin SQL Dump
-- version 5.2.1
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1
-- Generation Time: Nov 06, 2025 at 06:50 AM
-- Server version: 10.4.32-MariaDB
-- PHP Version: 8.2.12

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `project`
--

-- --------------------------------------------------------

--
-- Table structure for table `camera_location`
--

CREATE TABLE `camera_location` (
  `location_id` int(10) UNSIGNED NOT NULL,
  `location_name` varchar(100) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `camera_location`
--

INSERT INTO `camera_location` (`location_id`, `location_name`) VALUES
(2, 'Cafeteria'),
(3, 'Library'),
(1, 'Main Entrance'),
(173, 'กระต่าย'),
(155, 'กระถิน'),
(163, 'กระโจง'),
(21, 'บ้านนี้มีรัก'),
(5, 'บ้านปุ'),
(205, 'อาคาร 20'),
(4, 'อาคาร 7');

-- --------------------------------------------------------

--
-- Table structure for table `daily_detection`
--

CREATE TABLE `daily_detection` (
  `stat_id` int(11) NOT NULL,
  `stat_date` date NOT NULL,
  `total_with_mask` int(11) NOT NULL DEFAULT 0,
  `total_without_mask` int(11) NOT NULL DEFAULT 0,
  `total_mask` int(11) NOT NULL DEFAULT 0,
  `wear_mask_percent` decimal(5,2) NOT NULL DEFAULT 0.00,
  `location_id` int(10) UNSIGNED NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `daily_detection`
--

INSERT INTO `daily_detection` (`stat_id`, `stat_date`, `total_with_mask`, `total_without_mask`, `total_mask`, `wear_mask_percent`, `location_id`) VALUES
(593, '2025-10-16', 14, 31, 45, 32.61, 155),
(638, '2025-10-16', 8, 14, 22, 39.13, 2),
(660, '2025-10-16', 775, 20, 795, 97.49, 4),
(1973, '2025-10-29', 54, 122, 176, 30.51, 205),
(2149, '2025-10-31', 2, 2, 4, 40.00, 2),
(2153, '2025-11-06', 0, 1, 1, 0.00, 2);

-- --------------------------------------------------------

--
-- Table structure for table `no_mask_images`
--

CREATE TABLE `no_mask_images` (
  `img_id` int(11) NOT NULL,
  `capture_datetime` datetime NOT NULL,
  `image_name` varchar(255) NOT NULL,
  `location_id` int(10) UNSIGNED NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `no_mask_images`
--

INSERT INTO `no_mask_images` (`img_id`, `capture_datetime`, `image_name`, `location_id`) VALUES
(289, '2025-10-16 12:28:08', 'without_mask_2025-10-16_12-28-08.jpg', 2),
(290, '2025-10-16 12:36:43', 'without_mask_2025-10-16_12-36-43.jpg', 2),
(513, '2025-10-31 03:11:50', 'without_mask_2025-10-31_03-11-50.jpg', 2),
(514, '2025-10-31 03:53:14', 'without_mask_2025-10-31_03-53-14.jpg', 2),
(515, '2025-11-06 12:44:40', 'without_mask_2025-11-06_12-44-40.jpg', 2),
(291, '2025-10-16 13:45:07', 'without_mask_2025-10-16_13-45-07.jpg', 4),
(292, '2025-10-16 13:45:20', 'without_mask_2025-10-16_13-45-20.jpg', 4),
(293, '2025-10-16 13:47:45', 'without_mask_2025-10-16_13-47-45.jpg', 4),
(294, '2025-10-16 13:47:53', 'without_mask_2025-10-16_13-47-53.jpg', 4),
(295, '2025-10-16 13:48:32', 'without_mask_2025-10-16_13-48-32.jpg', 4),
(296, '2025-10-16 13:50:02', 'without_mask_2025-10-16_13-50-02.jpg', 4),
(297, '2025-10-16 13:50:24', 'without_mask_2025-10-16_13-50-24.jpg', 4),
(298, '2025-10-16 13:50:26', 'without_mask_2025-10-16_13-50-26.jpg', 4),
(299, '2025-10-16 13:50:29', 'without_mask_2025-10-16_13-50-29.jpg', 4),
(300, '2025-10-16 13:50:34', 'without_mask_2025-10-16_13-50-34.jpg', 4),
(301, '2025-10-16 13:50:38', 'without_mask_2025-10-16_13-50-38.jpg', 4),
(302, '2025-10-16 13:50:41', 'without_mask_2025-10-16_13-50-41.jpg', 4),
(303, '2025-10-16 13:50:44', 'without_mask_2025-10-16_13-50-44.jpg', 4),
(304, '2025-10-16 13:50:48', 'without_mask_2025-10-16_13-50-48.jpg', 4),
(261, '2025-10-16 02:36:16', 'without_mask_2025-10-16_02-36-16.jpg', 155),
(262, '2025-10-16 02:36:18', 'without_mask_2025-10-16_02-36-18.jpg', 155),
(263, '2025-10-16 02:36:20', 'without_mask_2025-10-16_02-36-20.jpg', 155),
(264, '2025-10-16 02:36:22', 'without_mask_2025-10-16_02-36-22.jpg', 155),
(265, '2025-10-16 02:36:24', 'without_mask_2025-10-16_02-36-24.jpg', 155),
(266, '2025-10-16 02:36:26', 'without_mask_2025-10-16_02-36-26.jpg', 155),
(267, '2025-10-16 02:36:28', 'without_mask_2025-10-16_02-36-28.jpg', 155),
(268, '2025-10-16 02:36:33', 'without_mask_2025-10-16_02-36-33.jpg', 155),
(269, '2025-10-16 02:36:37', 'without_mask_2025-10-16_02-36-37.jpg', 155),
(270, '2025-10-16 02:36:39', 'without_mask_2025-10-16_02-36-39.jpg', 155),
(271, '2025-10-16 02:36:42', 'without_mask_2025-10-16_02-36-42.jpg', 155),
(272, '2025-10-16 02:36:44', 'without_mask_2025-10-16_02-36-44.jpg', 155),
(273, '2025-10-16 02:36:46', 'without_mask_2025-10-16_02-36-46.jpg', 155),
(274, '2025-10-16 02:36:48', 'without_mask_2025-10-16_02-36-48.jpg', 155),
(275, '2025-10-16 02:36:50', 'without_mask_2025-10-16_02-36-50.jpg', 155),
(276, '2025-10-16 02:36:52', 'without_mask_2025-10-16_02-36-52.jpg', 155),
(277, '2025-10-16 02:38:32', 'without_mask_2025-10-16_02-38-32.jpg', 155),
(278, '2025-10-16 02:38:34', 'without_mask_2025-10-16_02-38-34.jpg', 155),
(279, '2025-10-16 02:38:36', 'without_mask_2025-10-16_02-38-36.jpg', 155),
(280, '2025-10-16 02:38:39', 'without_mask_2025-10-16_02-38-39.jpg', 155),
(281, '2025-10-16 02:38:41', 'without_mask_2025-10-16_02-38-41.jpg', 155),
(282, '2025-10-16 02:38:43', 'without_mask_2025-10-16_02-38-43.jpg', 155),
(283, '2025-10-16 02:38:45', 'without_mask_2025-10-16_02-38-45.jpg', 155),
(284, '2025-10-16 02:38:49', 'without_mask_2025-10-16_02-38-49.jpg', 155),
(285, '2025-10-16 02:38:51', 'without_mask_2025-10-16_02-38-51.jpg', 155),
(286, '2025-10-16 02:38:54', 'without_mask_2025-10-16_02-38-54.jpg', 155),
(287, '2025-10-16 02:38:56', 'without_mask_2025-10-16_02-38-56.jpg', 155),
(288, '2025-10-16 02:38:58', 'without_mask_2025-10-16_02-38-58.jpg', 155),
(424, '2025-10-29 11:52:45', 'without_mask_2025-10-29_11-52-45.jpg', 205),
(425, '2025-10-29 11:53:29', 'without_mask_2025-10-29_11-53-29.jpg', 205),
(426, '2025-10-29 11:53:36', 'without_mask_2025-10-29_11-53-36.jpg', 205),
(427, '2025-10-29 11:53:38', 'without_mask_2025-10-29_11-53-38.jpg', 205),
(428, '2025-10-29 11:53:41', 'without_mask_2025-10-29_11-53-41.jpg', 205),
(429, '2025-10-29 11:53:47', 'without_mask_2025-10-29_11-53-47.jpg', 205),
(430, '2025-10-29 11:53:54', 'without_mask_2025-10-29_11-53-54.jpg', 205),
(431, '2025-10-29 11:54:00', 'without_mask_2025-10-29_11-54-00.jpg', 205),
(432, '2025-10-29 11:54:08', 'without_mask_2025-10-29_11-54-08.jpg', 205),
(433, '2025-10-29 11:54:20', 'without_mask_2025-10-29_11-54-20.jpg', 205),
(434, '2025-10-29 11:54:25', 'without_mask_2025-10-29_11-54-25.jpg', 205),
(435, '2025-10-29 11:54:51', 'without_mask_2025-10-29_11-54-51.jpg', 205),
(436, '2025-10-29 11:55:20', 'without_mask_2025-10-29_11-55-20.jpg', 205),
(437, '2025-10-29 11:55:33', 'without_mask_2025-10-29_11-55-33.jpg', 205),
(438, '2025-10-29 11:55:45', 'without_mask_2025-10-29_11-55-45.jpg', 205),
(439, '2025-10-29 11:56:35', 'without_mask_2025-10-29_11-56-35.jpg', 205),
(440, '2025-10-29 12:00:01', 'without_mask_2025-10-29_12-00-01.jpg', 205),
(441, '2025-10-29 12:00:06', 'without_mask_2025-10-29_12-00-06.jpg', 205),
(442, '2025-10-29 12:01:33', 'without_mask_2025-10-29_12-01-33.jpg', 205),
(443, '2025-10-29 12:01:39', 'without_mask_2025-10-29_12-01-39.jpg', 205),
(444, '2025-10-29 12:01:48', 'without_mask_2025-10-29_12-01-48.jpg', 205),
(445, '2025-10-29 12:03:37', 'without_mask_2025-10-29_12-03-37.jpg', 205),
(446, '2025-10-29 12:04:16', 'without_mask_2025-10-29_12-04-16.jpg', 205),
(447, '2025-10-29 12:04:38', 'without_mask_2025-10-29_12-04-38.jpg', 205),
(448, '2025-10-29 12:04:42', 'without_mask_2025-10-29_12-04-42.jpg', 205),
(449, '2025-10-29 12:04:53', 'without_mask_2025-10-29_12-04-53.jpg', 205),
(450, '2025-10-29 12:04:59', 'without_mask_2025-10-29_12-04-59.jpg', 205),
(451, '2025-10-29 12:06:09', 'without_mask_2025-10-29_12-06-09.jpg', 205),
(452, '2025-10-29 12:09:31', 'without_mask_2025-10-29_12-09-31.jpg', 205),
(453, '2025-10-29 13:42:59', 'without_mask_2025-10-29_13-42-59.jpg', 205),
(454, '2025-10-29 13:43:12', 'without_mask_2025-10-29_13-43-12.jpg', 205),
(455, '2025-10-29 13:43:15', 'without_mask_2025-10-29_13-43-15.jpg', 205),
(456, '2025-10-29 13:43:19', 'without_mask_2025-10-29_13-43-19.jpg', 205),
(457, '2025-10-29 13:43:33', 'without_mask_2025-10-29_13-43-33.jpg', 205),
(458, '2025-10-29 13:43:38', 'without_mask_2025-10-29_13-43-38.jpg', 205),
(459, '2025-10-29 13:43:40', 'without_mask_2025-10-29_13-43-40.jpg', 205),
(460, '2025-10-29 13:43:45', 'without_mask_2025-10-29_13-43-45.jpg', 205),
(461, '2025-10-29 13:43:57', 'without_mask_2025-10-29_13-43-57.jpg', 205),
(462, '2025-10-29 13:44:06', 'without_mask_2025-10-29_13-44-06.jpg', 205),
(463, '2025-10-29 13:44:21', 'without_mask_2025-10-29_13-44-21.jpg', 205),
(464, '2025-10-29 13:44:26', 'without_mask_2025-10-29_13-44-26.jpg', 205),
(465, '2025-10-29 13:44:42', 'without_mask_2025-10-29_13-44-42.jpg', 205),
(466, '2025-10-29 13:44:45', 'without_mask_2025-10-29_13-44-45.jpg', 205),
(467, '2025-10-29 13:44:51', 'without_mask_2025-10-29_13-44-51.jpg', 205),
(468, '2025-10-29 13:44:54', 'without_mask_2025-10-29_13-44-54.jpg', 205),
(469, '2025-10-29 13:45:15', 'without_mask_2025-10-29_13-45-15.jpg', 205),
(470, '2025-10-29 13:45:19', 'without_mask_2025-10-29_13-45-19.jpg', 205),
(471, '2025-10-29 13:45:24', 'without_mask_2025-10-29_13-45-24.jpg', 205),
(472, '2025-10-29 13:45:26', 'without_mask_2025-10-29_13-45-26.jpg', 205),
(473, '2025-10-29 13:45:30', 'without_mask_2025-10-29_13-45-30.jpg', 205),
(474, '2025-10-29 13:45:34', 'without_mask_2025-10-29_13-45-34.jpg', 205),
(475, '2025-10-29 13:45:37', 'without_mask_2025-10-29_13-45-37.jpg', 205),
(476, '2025-10-29 13:45:48', 'without_mask_2025-10-29_13-45-48.jpg', 205),
(477, '2025-10-29 13:45:52', 'without_mask_2025-10-29_13-45-52.jpg', 205),
(478, '2025-10-29 13:45:54', 'without_mask_2025-10-29_13-45-54.jpg', 205),
(479, '2025-10-29 13:46:57', 'without_mask_2025-10-29_13-46-57.jpg', 205),
(480, '2025-10-29 13:47:16', 'without_mask_2025-10-29_13-47-16.jpg', 205),
(481, '2025-10-29 13:47:26', 'without_mask_2025-10-29_13-47-26.jpg', 205),
(482, '2025-10-29 13:47:41', 'without_mask_2025-10-29_13-47-41.jpg', 205),
(483, '2025-10-29 13:47:55', 'without_mask_2025-10-29_13-47-55.jpg', 205),
(484, '2025-10-29 13:48:37', 'without_mask_2025-10-29_13-48-37.jpg', 205),
(485, '2025-10-29 13:48:51', 'without_mask_2025-10-29_13-48-51.jpg', 205),
(486, '2025-10-29 13:48:55', 'without_mask_2025-10-29_13-48-55.jpg', 205),
(487, '2025-10-29 13:49:54', 'without_mask_2025-10-29_13-49-54.jpg', 205),
(488, '2025-10-29 13:50:12', 'without_mask_2025-10-29_13-50-12.jpg', 205),
(489, '2025-10-29 13:50:18', 'without_mask_2025-10-29_13-50-18.jpg', 205),
(490, '2025-10-29 13:51:21', 'without_mask_2025-10-29_13-51-21.jpg', 205),
(491, '2025-10-29 13:51:28', 'without_mask_2025-10-29_13-51-28.jpg', 205),
(492, '2025-10-29 13:53:57', 'without_mask_2025-10-29_13-53-57.jpg', 205),
(493, '2025-10-29 13:54:04', 'without_mask_2025-10-29_13-54-04.jpg', 205),
(494, '2025-10-29 13:54:15', 'without_mask_2025-10-29_13-54-15.jpg', 205),
(495, '2025-10-29 13:54:19', 'without_mask_2025-10-29_13-54-19.jpg', 205),
(496, '2025-10-29 13:54:31', 'without_mask_2025-10-29_13-54-31.jpg', 205),
(497, '2025-10-29 13:54:49', 'without_mask_2025-10-29_13-54-49.jpg', 205),
(498, '2025-10-29 13:55:17', 'without_mask_2025-10-29_13-55-17.jpg', 205),
(499, '2025-10-29 13:55:48', 'without_mask_2025-10-29_13-55-48.jpg', 205),
(500, '2025-10-29 13:55:51', 'without_mask_2025-10-29_13-55-51.jpg', 205),
(501, '2025-10-29 13:55:55', 'without_mask_2025-10-29_13-55-55.jpg', 205),
(502, '2025-10-29 13:56:21', 'without_mask_2025-10-29_13-56-21.jpg', 205),
(503, '2025-10-29 13:56:23', 'without_mask_2025-10-29_13-56-23.jpg', 205),
(504, '2025-10-29 13:57:27', 'without_mask_2025-10-29_13-57-27.jpg', 205),
(505, '2025-10-29 13:57:46', 'without_mask_2025-10-29_13-57-46.jpg', 205),
(506, '2025-10-29 13:58:16', 'without_mask_2025-10-29_13-58-16.jpg', 205),
(507, '2025-10-29 13:58:18', 'without_mask_2025-10-29_13-58-18.jpg', 205),
(508, '2025-10-29 13:58:21', 'without_mask_2025-10-29_13-58-21.jpg', 205),
(509, '2025-10-29 13:58:41', 'without_mask_2025-10-29_13-58-41.jpg', 205),
(510, '2025-10-29 13:58:44', 'without_mask_2025-10-29_13-58-44.jpg', 205),
(511, '2025-10-29 13:58:52', 'without_mask_2025-10-29_13-58-52.jpg', 205),
(512, '2025-10-29 13:58:56', 'without_mask_2025-10-29_13-58-56.jpg', 205);

--
-- Indexes for dumped tables
--

--
-- Indexes for table `camera_location`
--
ALTER TABLE `camera_location`
  ADD PRIMARY KEY (`location_id`),
  ADD UNIQUE KEY `uq_location_name` (`location_name`);

--
-- Indexes for table `daily_detection`
--
ALTER TABLE `daily_detection`
  ADD PRIMARY KEY (`stat_id`),
  ADD UNIQUE KEY `uq_dd_loc_time` (`location_id`,`stat_date`),
  ADD KEY `idx_dd_datetime` (`stat_date`);

--
-- Indexes for table `no_mask_images`
--
ALTER TABLE `no_mask_images`
  ADD PRIMARY KEY (`img_id`),
  ADD UNIQUE KEY `uq_img_loc_time_name` (`location_id`,`capture_datetime`,`image_name`),
  ADD KEY `idx_img_datetime` (`capture_datetime`);

--
-- AUTO_INCREMENT for dumped tables
--

--
-- AUTO_INCREMENT for table `camera_location`
--
ALTER TABLE `camera_location`
  MODIFY `location_id` int(10) UNSIGNED NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=213;

--
-- AUTO_INCREMENT for table `daily_detection`
--
ALTER TABLE `daily_detection`
  MODIFY `stat_id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=2154;

--
-- AUTO_INCREMENT for table `no_mask_images`
--
ALTER TABLE `no_mask_images`
  MODIFY `img_id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=516;

--
-- Constraints for dumped tables
--

--
-- Constraints for table `daily_detection`
--
ALTER TABLE `daily_detection`
  ADD CONSTRAINT `fk_dd_location` FOREIGN KEY (`location_id`) REFERENCES `camera_location` (`location_id`) ON UPDATE CASCADE;

--
-- Constraints for table `no_mask_images`
--
ALTER TABLE `no_mask_images`
  ADD CONSTRAINT `fk_img_location` FOREIGN KEY (`location_id`) REFERENCES `camera_location` (`location_id`) ON UPDATE CASCADE;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
