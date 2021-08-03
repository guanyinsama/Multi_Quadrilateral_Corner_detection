cmake_minimum_required(VERSION 2.8.3)
project(multiple_quadrilateral_corner_detection)
add_compile_options(-std=c++11)

find_package(PCL REQUIRED)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  pcl_ros
  cv_bridge
  sensor_msgs
  pcl_conversions
  message_generation
)

find_package(Qt5Widgets)
find_package(Qt5Core)
find_package( OpenCV REQUIRED )

catkin_package(
  CATKIN_DEPENDS std_msgs sensor_msgs message_runtime
)

include_directories(include ${catkin_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${Qt5Widgets_INCLUDE_DIRS}
${Qt5Core_INCLUDE_DIRS})
SET(CMAKE_CXX_FLAGS "-std=c++11 -O2 -g -Wall ${CMAKE_CXX_FLAGS}")

include_directories(${PCL_INCLUDE_DIRS})

link_directories(${PCL_LIBRARY_DIRS})

add_definitions(${PCL_DEFINITIONS})

list(REMOVE_ITEM PCL_LIBRARIES "vtkproj4")

add_executable(corner_detection
  src/main.cpp
)


target_link_libraries(corner_detection
  Qt5::Core
  Qt5::Widgets
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
  ${PCL_LIBRARIES}
)

add_dependencies(corner_detection ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

add_executable(calibration_steps
        src/main.cpp
        )


target_link_libraries(calibration_steps
        Qt5::Core
        Qt5::Widgets
        ${catkin_LIBRARIES}
        ${OpenCV_LIBS}
        ${PCL_LIBRARIES}
        )

add_dependencies(calibration_steps ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
